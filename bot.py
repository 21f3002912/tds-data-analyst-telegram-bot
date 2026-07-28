import asyncio
import json
import os

from dotenv import load_dotenv
from google import genai
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters


load_dotenv(".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing from .env")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")


client = genai.Client(api_key=GEMINI_API_KEY)


# Store recent conversation history separately for each Telegram chat.
# This lets the bot answer multi-turn questions.
conversation_history: dict[int, list[str]] = {}

# Keep only the most recent messages so context does not grow forever.
MAX_HISTORY = 10


def ask_gemini(messages: list[str]) -> str:
    """Send the conversation to Gemini and return its JSON response."""

    conversation = "\n\n".join(
        f"Message {i + 1}:\n{message}"
        for i, message in enumerate(messages)
    )

    prompt = f"""
You are a data-analysis assistant.

You may receive a multi-turn conversation.
The final message is the question you must answer.
Use earlier messages as context whenever necessary.

Answer the user's question accurately.

Return ONLY one valid JSON object in exactly this outer format:
{{"answer": <answer>, "log_url": "https://example.com/run.jsonl"}}

The value of "answer" must follow the shape requested by the user.
It may be a string, number, list, object, boolean, or other valid JSON value.

Do not use Markdown.
Do not use code fences.
Do not write explanations before or after the JSON.

Conversation:
{conversation}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    return response.text.strip()


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle each incoming Telegram text message."""

    if update.message is None or update.message.text is None:
        return

    try:
        chat = update.effective_chat

        if chat is None:
            raise RuntimeError("Could not determine Telegram chat")

        chat_id = chat.id

        # Get this chat's existing conversation or create a new one.
        history = conversation_history.setdefault(chat_id, [])

        # Add the newest user message.
        history.append(update.message.text)

        # Prevent history from growing indefinitely.
        if len(history) > MAX_HISTORY:
            del history[:-MAX_HISTORY]

        # Use a copy so another incoming message cannot modify the list
        # while Gemini is processing it.
        result = await asyncio.to_thread(
            ask_gemini,
            history.copy(),
        )

        # Gemini must return valid JSON.
        parsed = json.loads(result)

        if not isinstance(parsed, dict):
            raise ValueError("Gemini response is not a JSON object")

        if "answer" not in parsed:
            raise ValueError('Gemini response is missing "answer"')

        if "log_url" not in parsed:
            raise ValueError('Gemini response is missing "log_url"')

        # Ensure Telegram receives exactly one compact JSON object.
        reply = json.dumps(
            parsed,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        await update.message.reply_text(reply)

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")

        # Temporary development fallback.
        fallback = {
            "answer": None,
            "log_url": "https://example.com/run.jsonl",
        }

        await update.message.reply_text(
            json.dumps(
                fallback,
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

    print("Telegram + Gemini bot is running...")
    application.run_polling()


if __name__ == "__main__":
    main()
