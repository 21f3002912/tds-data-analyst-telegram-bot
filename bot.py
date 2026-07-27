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


def ask_gemini(question: str) -> str:
    prompt = f"""
You are a data-analysis assistant.

Answer the user's question.

Return ONLY one valid JSON object in exactly this outer format:
{{"answer": <answer>, "log_url": "https://example.com/run.jsonl"}}

Do not use Markdown.
Do not use code fences.
Do not write any text before or after the JSON.

User question:
{question}
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
    if update.message is None or update.message.text is None:
        return

    try:
        result = await asyncio.to_thread(
            ask_gemini,
            update.message.text,
        )

        # Validate that Gemini actually returned one JSON object.
        parsed = json.loads(result)

        if not isinstance(parsed, dict):
            raise ValueError("Gemini response is not a JSON object")

        if "answer" not in parsed or "log_url" not in parsed:
            raise ValueError("Required JSON keys are missing")

        await update.message.reply_text(
            json.dumps(parsed, separators=(",", ":"))
        )

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")

        # Temporary development response.
        await update.message.reply_text(
            json.dumps(
                {
                    "answer": None,
                    "log_url": "https://example.com/run.jsonl",
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

    print("Telegram + Gemini bot is running...")
    application.run_polling()


if __name__ == "__main__":
    main()
