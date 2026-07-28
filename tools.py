import io
import ipaddress
import socket
from urllib.parse import urlparse

import pandas as pd
import requests


MAX_DOWNLOAD_BYTES = 5_000_000
TIMEOUT_SECONDS = 20


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("URL has no hostname.")

    addresses = socket.getaddrinfo(parsed.hostname, None)

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
        ):
            raise ValueError("Private or local network URLs are not allowed.")


def fetch_url(url: str) -> dict:
    """Fetch a public HTTP/HTTPS resource."""

    _validate_public_url(url)

    response = requests.get(
        url,
        timeout=TIMEOUT_SECONDS,
        headers={"User-Agent": "TDS-Data-Analyst-Bot/1.0"},
        stream=True,
    )

    response.raise_for_status()

    content_length = response.headers.get("content-length")

    if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Resource is larger than the allowed download size.")

    data = bytearray()

    for chunk in response.iter_content(chunk_size=65536):
        data.extend(chunk)

        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Resource exceeded the allowed download size.")

    encoding = response.encoding or "utf-8"
    text = bytes(data).decode(encoding, errors="replace")

    return {
        "url": response.url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "size_bytes": len(data),
        "content": text,
    }


def analyse_csv(
    url: str,
    operation: str,
    column: str = "",
    filter_column: str = "",
    filter_value: str = "",
) -> dict:
    """
    Analyse a public CSV using pandas.

    Supported operations:
    - row_count
    - sum
    - mean
    - min
    - max

    Optionally filter rows first using filter_column and filter_value.
    """

    result = fetch_url(url)

    df = pd.read_csv(io.StringIO(result["content"]))

    # Normalise column names.
    df.columns = [str(c).strip() for c in df.columns]

    if filter_column:
        if filter_column not in df.columns:
            raise ValueError(
                f"Filter column '{filter_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        df = df[
            df[filter_column].astype(str).str.strip()
            == str(filter_value).strip()
        ]

    operation = operation.lower().strip()

    if operation == "row_count":
        value = int(len(df))

    else:
        if not column:
            raise ValueError(
                f"Operation '{operation}' requires a column."
            )

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        numeric = pd.to_numeric(df[column], errors="coerce")

        if operation == "sum":
            value = float(numeric.sum())

        elif operation == "mean":
            value = float(numeric.mean())

        elif operation == "min":
            value = float(numeric.min())

        elif operation == "max":
            value = float(numeric.max())

        else:
            raise ValueError(
                "Unsupported operation. Use row_count, sum, mean, min, or max."
            )

    return {
        "operation": operation,
        "rows_after_filter": int(len(df)),
        "result": value,
    }