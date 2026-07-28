import asyncio
import os
from pathlib import Path

import uvicorn
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, Response, request
from telegram import Update

from bot import application


PORT = int(os.getenv("PORT", "8000"))
WEBHOOK_URL = os.getenv(
    "WEBHOOK_URL",
    "https://tds-data-analyst-telegram-bot-ym39.onrender.com/telegram",
)

LOG_FILE = Path("run.jsonl")

flask_app = Flask(__name__)


@flask_app.get("/")
def health():
    return {"status": "ok"}


@flask_app.get("/run.jsonl")
def run_log():
    if not LOG_FILE.exists():
        return Response(
            "",
            status=200,
            mimetype="application/x-ndjson",
        )

    return Response(
        LOG_FILE.read_text(encoding="utf-8"),
        status=200,
        mimetype="application/x-ndjson",
    )


@flask_app.post("/telegram")
async def telegram_webhook():
    data = request.get_json(force=True)

    await application.update_queue.put(
        Update.de_json(data=data, bot=application.bot)
    )

    return "", 200


asgi_app = WsgiToAsgi(flask_app)


async def main() -> None:
    await application.bot.set_webhook(
        url=WEBHOOK_URL,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )

    config = uvicorn.Config(
        app=asgi_app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )

    server = uvicorn.Server(config)

    print(f"Webhook URL: {WEBHOOK_URL}")
    print(f"Log URL: {os.getenv('LOG_URL')}")

    async with application:
        await application.start()

        try:
            await server.serve()
        finally:
            await application.stop()


if __name__ == "__main__":
    asyncio.run(main())