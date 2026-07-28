import io
import ipaddress
import json
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
        allow_redirects=True,
    )

    response.raise_for_status()

    # Validate the final URL too, in case the original URL redirected.
    _validate_public_url(response.url)

    content_length = response.headers.get("content-length")

    if content_length:
        try:
            if int(content_length) > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    "Resource is larger than the allowed download size."
                )
        except ValueError as exc:
            if "larger than" in str(exc):
                raise

    data = bytearray()

    for chunk in response.iter_content(chunk_size=65536):
        data.extend(chunk)

        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ValueError(
                "Resource exceeded the allowed download size."
            )

    encoding = response.encoding or "utf-8"
    text = bytes(data).decode(encoding, errors="replace")

    return {
        "url": response.url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "size_bytes": len(data),
        "content": text,
    }


def _python_value(value):
    """Convert pandas/numpy values into JSON-friendly Python values."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def _load_dataframe(url: str) -> pd.DataFrame:
    """Load CSV or JSON tabular data from a public URL."""

    result = fetch_url(url)
    content = result["content"]
    content_type = result["content_type"].lower()

    looks_json = (
        "json" in content_type
        or url.lower().split("?")[0].endswith(".json")
        or content.lstrip().startswith(("{", "["))
    )

    if looks_json:
        obj = json.loads(content)

        if isinstance(obj, list):
            df = pd.json_normalize(obj)

        elif isinstance(obj, dict):
            # Common API structures:
            # {"data": [...]}, {"results": [...]}, {"records": [...]}
            candidate = None

            for key in ("data", "results", "records", "items"):
                if isinstance(obj.get(key), list):
                    candidate = obj[key]
                    break

            if candidate is not None:
                df = pd.json_normalize(candidate)
            else:
                df = pd.json_normalize(obj)

        else:
            raise ValueError("JSON resource is not tabular.")

    else:
        df = pd.read_csv(io.StringIO(content))

    df.columns = [str(c).strip() for c in df.columns]

    return df


def analyse_csv(
    url: str,
    operation: str,
    column: str = "",
    filter_column: str = "",
    filter_value: str = "",
    group_by: str = "",
    sort_by: str = "",
    ascending: bool = True,
    limit: int = 10,
    second_column: str = "",
) -> dict:
    """
    Analyse public CSV or JSON tabular data using pandas.

    Supported operations:
    - columns
    - row_count
    - unique_count
    - unique_values
    - sum
    - mean
    - median
    - min
    - max
    - std
    - variance
    - correlation
    - value_counts
    - group_sum
    - group_mean
    - group_count
    - top
    - bottom

    filter_column/filter_value optionally apply an equality filter first.

    group_by is used by group_sum, group_mean and group_count.

    second_column is required for correlation.

    sort_by can be used by top/bottom. If omitted, column is used.

    limit controls the maximum number of returned rows or values.
    """

    df = _load_dataframe(url)

    operation = operation.lower().strip()
    column = column.strip()
    filter_column = filter_column.strip()
    group_by = group_by.strip()
    sort_by = sort_by.strip()
    second_column = second_column.strip()

    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit = 10

    if filter_column:
        if filter_column not in df.columns:
            raise ValueError(
                f"Filter column '{filter_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        series = df[filter_column]

        # First try a numeric comparison when both sides are numeric.
        numeric_series = pd.to_numeric(series, errors="coerce")

        try:
            numeric_filter = float(filter_value)
            numeric_mask = numeric_series == numeric_filter

            if numeric_mask.any():
                df = df[numeric_mask]
            else:
                df = df[
                    series.astype(str).str.strip()
                    == str(filter_value).strip()
                ]

        except (TypeError, ValueError):
            df = df[
                series.astype(str).str.strip()
                == str(filter_value).strip()
            ]

    if operation == "columns":
        return {
            "operation": operation,
            "rows_after_filter": int(len(df)),
            "columns": list(df.columns),
        }

    if operation == "row_count":
        value = int(len(df))

        return {
            "operation": operation,
            "rows_after_filter": int(len(df)),
            "result": value,
        }

    if operation in {
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "variance",
        "unique_count",
        "unique_values",
        "value_counts",
    }:
        if not column:
            raise ValueError(
                f"Operation '{operation}' requires a column."
            )

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    if operation == "unique_count":
        value = int(df[column].nunique(dropna=True))

    elif operation == "unique_values":
        values = [
            _python_value(v)
            for v in df[column].dropna().unique()[:limit]
        ]

        return {
            "operation": operation,
            "rows_after_filter": int(len(df)),
            "column": column,
            "result": values,
        }

    elif operation == "value_counts":
        counts = (
            df[column]
            .value_counts(dropna=False)
            .head(limit)
        )

        values = [
            {
                "value": _python_value(index),
                "count": int(count),
            }
            for index, count in counts.items()
        ]

        return {
            "operation": operation,
            "rows_after_filter": int(len(df)),
            "column": column,
            "result": values,
        }

    elif operation in {
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "variance",
    }:
        numeric = pd.to_numeric(df[column], errors="coerce").dropna()

        if numeric.empty:
            raise ValueError(
                f"Column '{column}' has no numeric values."
            )

        if operation == "sum":
            value = float(numeric.sum())

        elif operation == "mean":
            value = float(numeric.mean())

        elif operation == "median":
            value = float(numeric.median())

        elif operation == "min":
            value = float(numeric.min())

        elif operation == "max":
            value = float(numeric.max())

        elif operation == "std":
            value = float(numeric.std())

        else:
            value = float(numeric.var())

    elif operation == "correlation":
        if not column or not second_column:
            raise ValueError(
                "correlation requires column and second_column."
            )

        for name in (column, second_column):
            if name not in df.columns:
                raise ValueError(
                    f"Column '{name}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

        x = pd.to_numeric(df[column], errors="coerce")
        y = pd.to_numeric(df[second_column], errors="coerce")

        valid = x.notna() & y.notna()

        if valid.sum() < 2:
            raise ValueError(
                "Not enough numeric rows to calculate correlation."
            )

        value = float(x[valid].corr(y[valid]))

    elif operation in {"group_sum", "group_mean", "group_count"}:
        if not group_by:
            raise ValueError(
                f"Operation '{operation}' requires group_by."
            )

        if group_by not in df.columns:
            raise ValueError(
                f"Group column '{group_by}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if operation == "group_count":
            grouped = (
                df.groupby(group_by, dropna=False)
                .size()
                .sort_values(ascending=False)
                .head(limit)
            )

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

            temp = df[[group_by, column]].copy()
            temp[column] = pd.to_numeric(
                temp[column],
                errors="coerce",
            )

            grouped_obj = temp.groupby(
                group_by,
                dropna=False,
            )[column]

            if operation == "group_sum":
                grouped = grouped_obj.sum()
            else:
                grouped = grouped_obj.mean()

            grouped = grouped.sort_values(
                ascending=False
            ).head(limit)

        values = [
            {
                "group": _python_value(index),
                "value": _python_value(value),
            }
            for index, value in grouped.items()
        ]

        return {
            "operation": operation,
            "rows_after_filter": int(len(df)),
            "group_by": group_by,
            "column": column or None,
            "result": values,
        }

    elif operation in {"top", "bottom"}:
        target = sort_by or column

        if not target:
            raise ValueError(
                f"Operation '{operation}' requires sort_by or column."
            )

        if target not in df.columns:
            raise ValueError(
                f"Sort column '{target}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        temp = df.copy()

        numeric = pd.to_numeric(temp[target], errors="coerce")

        if numeric.notna().sum() > 0:
            temp["_sort_value"] = numeric
            target_for_sort = "_sort_value"
        else:
            target_for_sort = target

        # top = largest first; bottom = smallest first.
        sort_ascending = operation == "bottom"

        temp = temp.sort_values(
            target_for_sort,
            ascending=sort_ascending,
            na_position="last",
        ).head(limit)

        if "_sort_value" in temp.columns:
            temp = temp.drop(columns=["_sort_value"])

        records = []

        for record in temp.to_dict(orient="records"):
            records.append(
                {
                    key: _python_value(value)
                    for key, value in record.items()
                }
            )

        return {
            "operation": operation,
            "rows_after_filter": int(len(df)),
            "sorted_by": target,
            "result": records,
        }

    else:
        raise ValueError(
            "Unsupported operation. Use columns, row_count, "
            "unique_count, unique_values, sum, mean, median, min, "
            "max, std, variance, correlation, value_counts, "
            "group_sum, group_mean, group_count, top, or bottom."
        )

    return {
        "operation": operation,
        "rows_after_filter": int(len(df)),
        "column": column or None,
        "result": _python_value(value),
    }