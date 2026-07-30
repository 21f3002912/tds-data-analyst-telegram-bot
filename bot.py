import asyncio
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

from run_logger import log_event
from tools import analyse_csv, fetch_url


load_dotenv(".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LOG_URL = os.getenv(
    "LOG_URL",
    "http://127.0.0.1:8000/run.jsonl",
)

if not TELEGRAM_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is missing from .env"
    )

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from .env"
    )


client = genai.Client(api_key=GEMINI_API_KEY)

conversation_history: dict[int, list[str]] = {}
MAX_HISTORY = 10

# Primary model plus fallback model.
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]

RETRIES_PER_MODEL = 3


def get_public_url(url: str) -> dict:
    """Fetch a public HTTP/HTTPS resource for analysis."""

    log_event(
        "tool_call",
        data={
            "tool": "get_public_url",
            "url": url,
        },
    )

    result = fetch_url(url)

    # Prevent extremely large resources from being returned
    # directly to the model.
    content = result["content"][:200_000]

    tool_result = {
        "url": result["url"],
        "status_code": result["status_code"],
        "content_type": result["content_type"],
        "size_bytes": result["size_bytes"],
        "content": content,
    }

    log_event(
        "tool_result",
        data={
            "tool": "get_public_url",
            "url": result["url"],
            "status_code": result["status_code"],
            "size_bytes": result["size_bytes"],
        },
    )

    return tool_result


def ask_gemini(messages: list[str]) -> str:
    conversation = "\n\n".join(
        f"Message {i + 1}:\n{message}"
        for i, message in enumerate(messages)
    )

    prompt = f"""
You are a data-analysis agent.

The final message is the question you must answer.
Earlier messages may contain important context.

When a question provides a public URL containing data, use the
get_public_url tool to retrieve the actual resource before answering.
Do not invent values that should be obtained from supplied data.

When the question provides a public CSV or JSON dataset, prefer
analyse_csv for deterministic analysis rather than calculating values
yourself.

analyse_csv supports:
columns, row_count, unique_count, unique_values, sum, mean, median,
min, max, std, variance, correlation, value_counts, group_sum,
group_mean, group_count, top, and bottom.

It also supports equality filtering, grouping, sorting, limits, and a
second numeric column for correlation.

If you do not know the dataset's column names or structure, first use
analyse_csv with operation="columns" or use get_public_url to inspect
the resource. Then perform the required analysis with analyse_csv.

For aggregation, filtering, grouping, ranking, counting, descriptive
statistics, and correlation, use analyse_csv whenever it can answer
the question. Do not estimate or manually calculate a result that the
tool can compute.

Use get_public_url when you need to inspect raw contents or when
analyse_csv cannot perform the requested operation.

Pay close attention to the requested answer type and shape. Return a
float when a float is requested, an integer when an integer is
requested, and preserve requested strings, arrays, or objects exactly.

Perform the required analysis carefully.

Return ONLY one valid JSON object with exactly these outer keys:

{{"answer": <answer>, "log_url": "{LOG_URL}"}}

The value of "answer" must follow exactly the shape requested by the
user.

Always set "log_url" to exactly:
{LOG_URL}

Do not use Markdown.
Do not use code fences.
Do not add explanations outside the JSON.

Conversation:
{conversation}
"""

    last_error = None

    for model in GEMINI_MODELS:

        for attempt in range(RETRIES_PER_MODEL):

            try:
                print(
                    f"Trying {model}, "
                    f"attempt {attempt + 1}/"
                    f"{RETRIES_PER_MODEL}",
                    flush=True,
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "tools": [
                            get_public_url,
                            analyse_csv,
                        ],
                    },
                )

                if not response.text:
                    raise RuntimeError(
                        f"{model} returned an empty response"
                    )

                print(
                    f"SUCCESS using {model}",
                    flush=True,
                )

                print(
                    f"RAW GEMINI RESPONSE: "
                    f"{response.text!r}",
                    flush=True,
                )

                return response.text.strip()

            except Exception as exc:
                last_error = exc

                error_text = str(exc).lower()

                print(
                    f"{model} attempt "
                    f"{attempt + 1} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

                retryable = any(
                    marker in error_text
                    for marker in [
                        "503",
                        "unavailable",
                        "high demand",
                        "429",
                        "resource_exhausted",
                        "rate limit",
                        "timeout",
                        "timed out",
                    ]
                )

                # Programming/configuration/tool errors should
                # not be hidden by repeatedly trying models.
                if not retryable:
                    raise

                if attempt < RETRIES_PER_MODEL - 1:
                    wait_seconds = 2 ** attempt

                    print(
                        f"Retrying {model} in "
                        f"{wait_seconds}s...",
                        flush=True,
                    )

                    time.sleep(wait_seconds)

        print(
            f"{model} unavailable after retries. "
            f"Trying fallback model...",
            flush=True,
        )

    raise RuntimeError(
        f"All Gemini models failed: {last_error}"
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if (
        update.message is None
        or update.message.text is None
    ):
        return

    chat = update.effective_chat

    if chat is None:
        return

    chat_id = chat.id
    question = update.message.text
    started = time.monotonic()

    log_event(
        "run_start",
        chat_id=chat_id,
        data={
            "message": question,
        },
    )

    try:
        history = conversation_history.setdefault(
            chat_id,
            [],
        )

        history.append(question)

        if len(history) > MAX_HISTORY:
            del history[:-MAX_HISTORY]

        result = await asyncio.to_thread(
            ask_gemini,
            history.copy(),
        )

        print(
            f"RESULT BEFORE JSON PARSE: {result!r}",
            flush=True,
        )

        parsed = json.loads(result)

        if not isinstance(parsed, dict):
            raise ValueError(
                "Gemini response is not a JSON object"
            )

        if "answer" not in parsed:
            raise ValueError(
                'Required key "answer" is missing'
            )

        # Python, rather than Gemini, controls
        # the authoritative log URL.
        parsed["log_url"] = LOG_URL

        reply = json.dumps(
            parsed,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        elapsed = round(
            time.monotonic() - started,
            3,
        )

        log_event(
            "run_complete",
            chat_id=chat_id,
            data={
                "answer": parsed["answer"],
                "log_url": LOG_URL,
                "elapsed_seconds": elapsed,
            },
        )

        print(
            f"SUCCESS: answer="
            f"{parsed['answer']!r}",
            flush=True,
        )

        await update.message.reply_text(reply)

    except Exception as exc:
        elapsed = round(
            time.monotonic() - started,
            3,
        )

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        log_event(
            "run_error",
            chat_id=chat_id,
            data={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed,
            },
        )

        print(
            f"ERROR: {error_message}",
            flush=True,
        )

        # TEMPORARY DEBUGGING:
        # Return the real exception instead of answer:null.
        # Once deployment is verified, we can change this
        # back to a cleaner production failure response.
        await update.message.reply_text(
            json.dumps(
                {
                    "answer": error_message,
                    "log_url": LOG_URL,
                },
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )


application = (
    Application.builder()
    .token(TELEGRAM_TOKEN)
    .updater(None)
    .build()
)

application.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message,
    )
)