from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...models import CandleRequest
from .constants import DEFAULT_USER_AGENT, WS_URL
from .credentials import TradingViewCredentials, resolve_credentials
from .protocol import cookie_header_value, encode_message, encode_raw, generate_session, split_payloads
from .timeframes import backfill_chunk_size, estimate_bar_count, max_backfill_requests, normalize_interval, to_timestamp


class ChartSession:
    """Manages a single TradingView WebSocket chart-session download."""

    def __init__(self, request: CandleRequest) -> None:
        self._request = request

    def download(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        try:
            import websocket  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "websocket-client is required for TradingView session downloads. "
                "Install dependencies from requirements.txt or use --csv."
            ) from error

        credentials = resolve_credentials()
        interval = normalize_interval(self._request.timeframe)
        bars_to_request = estimate_bar_count(
            self._request.timeframe, self._request.start, self._request.end,
        )

        headers = self._build_headers(credentials)
        ws = websocket.create_connection(WS_URL, header=headers, timeout=30)
        try:
            return self._run_session(ws, credentials, interval, bars_to_request)
        except BaseException:
            ws.close()
            raise

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _run_session(
        self,
        ws: Any,
        credentials: TradingViewCredentials,
        interval: str,
        bars_to_request: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        self._handle_initial_heartbeat(ws)
        chart_session = generate_session("cs")
        quote_session = generate_session("qs")
        symbol = f"{self._request.exchange}:{self._request.symbol}"
        resolved_symbol = json.dumps(
            {
                "symbol": symbol,
                "adjustment": self._request.adjustment,
                "session": self._request.session,
            },
            separators=(",", ":"),
        )

        self._setup_sessions(ws, credentials, chart_session, quote_session, symbol, resolved_symbol, interval, bars_to_request)
        return self._receive_loop(ws, credentials, chart_session)

    def _setup_sessions(
        self,
        ws: Any,
        credentials: TradingViewCredentials,
        chart_session: str,
        quote_session: str,
        symbol: str,
        resolved_symbol: str,
        interval: str,
        bars_to_request: int,
    ) -> None:
        if credentials.auth_token != "unauthorized_user_token":
            self._send(ws, "set_auth_token", [credentials.auth_token])
        self._send(ws, "chart_create_session", [chart_session, ""])
        self._send(ws, "quote_create_session", [quote_session])
        self._send(
            ws,
            "quote_set_fields",
            [
                quote_session,
                "ch", "chp", "current_session", "description", "exchange",
                "format", "fractional", "is_tradable", "lp", "lp_time",
                "minmov", "pricescale", "pro_name", "short_name", "type", "volume",
            ],
        )
        self._send(ws, "quote_add_symbols", [quote_session, symbol, {"flags": ["force_permission"]}])
        self._send(ws, "quote_fast_symbols", [quote_session, symbol])
        self._send(ws, "resolve_symbol", [chart_session, "symbol_1", f"={resolved_symbol}"])
        self._send(ws, "create_series", [chart_session, "s1", "s1", "symbol_1", interval, bars_to_request])
        self._send(ws, "switch_timezone", [chart_session, "exchange"])

    def _receive_loop(
        self,
        ws: Any,
        credentials: TradingViewCredentials,
        chart_session: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        backfill_count = 0
        previous_oldest_time: int | None = None

        while True:
            raw = ws.recv()
            for payload in split_payloads(raw):
                if payload.isdigit():
                    ws.send(encode_raw(payload))
                    continue
                try:
                    message = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                payloads.append(message)

                if not (isinstance(message, dict) and message.get("m") == "series_completed"):
                    continue

                dataframe = _extract_dataframe(payloads)
                metadata = _extract_history_metadata(payloads, self._request, credentials, backfill_count)

                if dataframe.empty:
                    ws.close()
                    raise RuntimeError(
                        "TradingView did not return candle data for this request. "
                        "Check your auth token, exchange permissions, or use --csv."
                    )

                oldest_time = int(dataframe["time"].min())
                if _should_request_more(self._request, dataframe, backfill_count, previous_oldest_time):
                    previous_oldest_time = oldest_time
                    backfill_count += 1
                    self._send(
                        ws,
                        "request_more_data",
                        [chart_session, "s1", backfill_chunk_size(self._request.timeframe)],
                    )
                    continue

                ws.close()
                return dataframe, metadata

    # ------------------------------------------------------------------
    # Transport helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_headers(credentials: TradingViewCredentials) -> list[str]:
        headers = [
            "Origin: https://www.tradingview.com",
            f"User-Agent: {DEFAULT_USER_AGENT}",
        ]
        header_cookies = cookie_header_value(credentials.cookies)
        if header_cookies:
            headers.append(f"Cookie: {header_cookies}")
        elif credentials.session_id:
            headers.append(f"Cookie: sessionid={credentials.session_id}")
        return headers

    @staticmethod
    def _send(ws: Any, function: str, parameters: list[Any]) -> None:
        ws.send(encode_message(function, parameters))

    @staticmethod
    def _handle_initial_heartbeat(ws: Any) -> None:
        raw = ws.recv()
        for payload in split_payloads(raw):
            if payload.isdigit():
                ws.send(encode_raw(payload))


# ======================================================================
# Pure extraction helpers (module-level, no state needed)
# ======================================================================

def _extract_dataframe(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    bars: list[list[float]] = []
    for payload in payloads:
        if payload.get("m") != "timescale_update":
            continue
        for item in _find_series_points(payload):
            values = item.get("v")
            if not isinstance(values, list) or len(values) < 5:
                continue
            bars.append(values)

    if not bars:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(bars)
    columns = ["time", "open", "high", "low", "close", "volume"]
    # Pad missing columns with NaN if TradingView returned fewer fields.
    frame = frame.reindex(columns=range(len(columns)))
    frame.columns = columns
    frame = frame.dropna(subset=["time", "open", "high", "low", "close"])
    frame["time"] = frame["time"].astype("int64")
    for column in ["open", "high", "low", "close"]:
        frame[column] = frame[column].astype("float64")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return frame


def _extract_history_metadata(
    payloads: list[dict[str, Any]],
    request: CandleRequest,
    credentials: TradingViewCredentials,
    backfill_requests: int,
) -> dict[str, Any]:
    requested_symbol = f"{request.exchange}:{request.symbol}"
    symbol_data: dict[str, Any] = {}
    for payload in payloads:
        if payload.get("m") != "symbol_resolved":
            continue
        parameters = payload.get("p")
        if not isinstance(parameters, list) or len(parameters) < 3 or not isinstance(parameters[2], dict):
            continue
        symbol_data = parameters[2]
        break

    resolved_full_name = symbol_data.get("full_name")
    resolved_pro_name = symbol_data.get("pro_name")
    return {
        "source": "tradingview-websocket",
        "requested_symbol": requested_symbol,
        "resolved_full_name": resolved_full_name,
        "resolved_pro_name": resolved_pro_name,
        "exchange": symbol_data.get("exchange"),
        "description": symbol_data.get("description"),
        "type": symbol_data.get("type"),
        "session": symbol_data.get("session"),
        "source_id": symbol_data.get("source_id"),
        "provider_id": symbol_data.get("provider_id"),
        "pricescale": symbol_data.get("pricescale"),
        "has_session_id": credentials.session_id is not None,
        "has_auth_token": credentials.auth_token != "unauthorized_user_token",
        "credentials_source": credentials.source,
        "full_name_mismatch": resolved_full_name not in {None, requested_symbol},
        "pro_name_mismatch": resolved_pro_name not in {None, requested_symbol},
        "backfill_requests": backfill_requests,
    }


def _find_series_points(payload: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "s" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "v" in item:
                        results.append(item)
            else:
                results.extend(_find_series_points(value))
    elif isinstance(payload, list):
        for item in payload:
            results.extend(_find_series_points(item))
    return results


def _should_request_more(
    request: CandleRequest,
    dataframe: pd.DataFrame,
    backfill_count: int,
    previous_oldest_time: int | None,
) -> bool:
    if dataframe.empty:
        return False

    oldest_time = int(dataframe["time"].min())

    if previous_oldest_time is not None and oldest_time >= previous_oldest_time:
        return False

    start_ts = to_timestamp(request.start)
    if start_ts is not None and oldest_time <= start_ts:
        return False

    return backfill_count < max_backfill_requests(request.timeframe, request.start, request.end)
