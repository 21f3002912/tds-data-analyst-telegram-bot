import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_FILE = Path("run.jsonl")
_lock = threading.Lock()


def log_event(
    event: str,
    *,
    chat_id: int | None = None,
    data: Any = None,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }

    if chat_id is not None:
        record["chat_id"] = chat_id

    if data is not None:
        record["data"] = data

    line = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )

    with _lock:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")