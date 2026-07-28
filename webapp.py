from pathlib import Path

from flask import Flask, Response


app = Flask(__name__)

LOG_FILE = Path("run.jsonl")


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/run.jsonl")
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


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
    )