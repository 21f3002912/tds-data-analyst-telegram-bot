import asyncio
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from run_logger import log_event
from tools import analyse_csv, fetch_url


load_dotenv(".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Temporary until deployment gives us a public URL.
LOG_URL = os.getenv("LOG_URL", "http://127.0.0.1:8000/run.jsonl")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing from .env")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")


client = genai.Client(api_key=GEMINI_API_KEY)

conversation_history: dict[int, list[str]] = {}
MAX_HISTORY = 10


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

When the data is in a public CSV and the requested calculation can be
performed using analyse_csv, use analyse_csv instead of calculating
values yourself.

Prefer tool-computed numerical results over estimating or manually
calculating from raw CSV text.

Use get_public_url when you need to inspect the contents, structure,
column names, or other information about a public resource.

Perform the required analysis carefully.

Return ONLY one valid JSON object with exactly these outer keys:

{{"answer": <answer>, "log_url": "{LOG_URL}"}}

The value of "answer" must follow exactly the shape requested by the user.

Always set "log_url" to exactly:
{LOG_URL}

Do not use Markdown.
Do not use code fences.
Do not add explanations outside the JSON.

Conversation:
{conversation}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config={
            "tools": [get_public_url, analyse_csv],
        },
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    return response.text.strip()


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if update.message is None or update.message.text is None:
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
        data={"message": question},
    )

    try:
        history = conversation_history.setdefault(chat_id, [])
        history.append(question)

        if len(history) > MAX_HISTORY:
            del history[:-MAX_HISTORY]

        result = await asyncio.to_thread(
            ask_gemini,
            history.copy(),
        )

        parsed = json.loads(result)

        if not isinstance(parsed, dict):
            raise ValueError("Gemini response is not a JSON object")

        if "answer" not in parsed:
            raise ValueError('Required key "answer" is missing')

        # Do not trust the model to preserve the URL.
        # Python sets the authoritative log URL.
        parsed["log_url"] = LOG_URL

        reply = json.dumps(
            parsed,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        elapsed = round(time.monotonic() - started, 3)

        log_event(
            "run_complete",
            chat_id=chat_id,
            data={
                "answer": parsed["answer"],
                "log_url": LOG_URL,
                "elapsed_seconds": elapsed,
            },
        )

        await update.message.reply_text(reply)

    except Exception as exc:
        elapsed = round(time.monotonic() - started, 3)

        log_event(
            "run_error",
            chat_id=chat_id,
            data={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": elapsed,
            },
        )

        print(f"ERROR: {type(exc).__name__}: {exc}")

        await update.message.reply_text(
            json.dumps(
                {
                    "answer": None,
                    "log_url": LOG_URL,
                },
                separators=(",", ":"),
            )
        )


def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    print("TDS Data Analyst Agent running...")
    print(f"Log URL: {LOG_URL}")

    application.run_polling()


if __name__ == "__main__":
    main()