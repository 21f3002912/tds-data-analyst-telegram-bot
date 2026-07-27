import json
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters


load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing from .env")


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Reply to every text message with valid JSON."""

    if update.message is None or update.message.text is None:
        return

    response = {
        "answer": {
            "received": update.message.text
        },
        "log_url": "https://example.com/run.jsonl",
    }

    # Compact JSON: no Markdown and no surrounding prose.
    await update.message.reply_text(
        json.dumps(response, separators=(",", ":"))
    )


def main() -> None:
    application = Application.builder().token(TOKEN).build()

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    print("Telegram bot is running...")
    application.run_polling()


if __name__ == "__main__":
    main()
