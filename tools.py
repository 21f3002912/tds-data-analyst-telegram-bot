import ipaddress
import socket
from urllib.parse import urlparse

import requests


MAX_DOWNLOAD_BYTES = 5_000_000
TIMEOUT_SECONDS = 20


def _validate_public_url(url: str) -> None:
    """Reject non-HTTP URLs and hosts resolving to private/local addresses."""

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
    """
    Fetch a public text/CSV/JSON resource and return its contents
    with basic metadata.
    """

    _validate_public_url(url)

    response = requests.get(
        url,
        timeout=TIMEOUT_SECONDS,
        headers={
            "User-Agent": "TDS-Data-Analyst-Bot/1.0",
        },
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
