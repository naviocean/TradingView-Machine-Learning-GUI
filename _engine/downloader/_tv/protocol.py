from __future__ import annotations

import json
import random
import re
import string
from typing import Any

# Pre-compiled once at import time — called on every WebSocket frame received.
_PAYLOAD_SPLIT_RE = re.compile(r"~m~\d+~m~")


def split_payloads(raw_message: str) -> list[str]:
    return [m for m in _PAYLOAD_SPLIT_RE.split(raw_message) if m]


def encode_message(function: str, parameters: list[Any]) -> str:
    payload = json.dumps({"m": function, "p": parameters}, separators=(",", ":"))
    return f"~m~{len(payload)}~m~{payload}"


def encode_raw(payload: str) -> str:
    return f"~m~{len(payload)}~m~{payload}"


def generate_session(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase, k=12))
    return f"{prefix}_{suffix}"


def cookie_header_value(cookies: dict[str, str]) -> str:
    return "; ".join(f"{name}={value}" for name, value in sorted(cookies.items()))
