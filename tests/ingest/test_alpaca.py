import datetime
import time
from collections.abc import Callable
from decimal import Decimal

import httpx
import pytest
from aiolimiter import AsyncLimiter
from pydantic import ValidationError

from horizon.ingest import alpaca
from horizon.ingest.alpaca import (
    MAX_ATTEMPTS,
    MAX_DELAY,
    _backoff_delay,
    _get_with_retry,
    fetch_daily_bars,
)

START = datetime.date(2026, 11, 2)
END = datetime.date(2026, 11, 6)


# High enough that no test ever waits on it. A fresh one per call, so tests
# never share a budget.
def fast_limiter() -> AsyncLimiter:
    return AsyncLimiter(1000, 1)


@pytest.mark.asyncio
async def test_fetch_parses_bars_exactly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "bars": {
                    "AAPL": [
                        {
                            "t": "2026-11-02T05:00:00Z",
                            "o": 342.67,
                            "h": 343.8,
                            "l": 335.82,
                            "c": 338.32,
                            "v": 64967561,
                            "vw": 338.972447,
                        },
                    ]
                },
                "next_page_token": None,
            },
        )

    # A client wired to the handler instead of the network.
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://test"
    )

    result = await fetch_daily_bars(client, fast_limiter(), ["AAPL"], START, END)
    [bar] = result["AAPL"]

    assert bar.close == Decimal("338.32")
    assert bar.vwap == Decimal("338.972447")  # exact: the Decimal parsing works
    assert bar.volume == 64967561
    assert bar.time == datetime.datetime(2026, 11, 2, 5, tzinfo=datetime.UTC)


@pytest.mark.asyncio
async def test_fetch_missing_vwap_is_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "bars": {
                    "AAPL": [
                        {
                            "t": "2026-11-02T05:00:00Z",
                            "o": 342.67,
                            "h": 343.8,
                            "l": 335.82,
                            "c": 338.32,
                            "v": 64967561,
                        },
                    ]
                },
                "next_page_token": None,
            },
        )

    # A client wired to the handler instead of the network.
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://test"
    )

    result = await fetch_daily_bars(client, fast_limiter(), ["AAPL"], START, END)
    [bar] = result["AAPL"]

    assert bar.close == Decimal("338.32")
    assert bar.vwap is None
    assert bar.volume == 64967561
    assert bar.time == datetime.datetime(2026, 11, 2, 5, tzinfo=datetime.UTC)


@pytest.mark.asyncio
async def test_fetch_error_status_raises() -> None:
    # Alpaca answers 401 when the keys are missing or wrong.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
        )

    # A client wired to the handler instead of the network.
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://test"
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await fetch_daily_bars(client, fast_limiter(), ["AAPL"], START, END)

    assert exc_info.value.response.status_code == 401


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://test"
    )


def bar(t: str, close: float) -> dict[str, object]:
    return {"t": t, "o": close, "h": close, "l": close, "c": close, "v": 1, "vw": close}


@pytest.mark.asyncio
async def test_fetch_unexpected_shape_raises() -> None:
    # "bars" should map symbol -> list; a bare list means the response format
    # changed, and nothing should be stored from it.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bars": [], "next_page_token": None})

    with pytest.raises(ValidationError):
        await fetch_daily_bars(
            make_client(handler), fast_limiter(), ["AAPL"], START, END
        )


@pytest.mark.asyncio
async def test_fetch_sends_raw_sip_daily_params() -> None:
    seen: list[httpx.QueryParams] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params)
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    await fetch_daily_bars(
        make_client(handler), fast_limiter(), ["AAPL", "MSFT"], START, END
    )

    [params] = seen
    assert params["symbols"] == "AAPL,MSFT"
    assert params["timeframe"] == "1Day"
    # Unadjusted prices are stored; splits and dividends are applied on read.
    assert params["adjustment"] == "raw"
    assert params["feed"] == "sip"
    assert params["start"] == "2026-11-02"
    assert params["end"] == "2026-11-06"
    assert "page_token" not in params


@pytest.mark.asyncio
async def test_fetch_merges_symbol_split_across_pages() -> None:
    # The page limit counts bars across all symbols, so one symbol's bars can
    # continue on the next page.
    pages = {
        None: {
            "bars": {
                "AAPL": [bar("2026-11-02T05:00:00Z", 1.0)],
                "MSFT": [bar("2026-11-02T05:00:00Z", 2.0)],
            },
            "next_page_token": "page-2",
        },
        "page-2": {
            "bars": {"MSFT": [bar("2026-11-03T05:00:00Z", 3.0)]},
            "next_page_token": None,
        },
    }
    tokens: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("page_token")
        tokens.append(token)
        return httpx.Response(200, json=pages[token])

    result = await fetch_daily_bars(
        make_client(handler), fast_limiter(), ["AAPL", "MSFT"], START, END
    )

    # The second request carried the first page's token, then the loop stopped.
    assert tokens == [None, "page-2"]
    assert [b.close for b in result["AAPL"]] == [Decimal("1.0")]
    assert [b.close for b in result["MSFT"]] == [Decimal("2.0"), Decimal("3.0")]


# --- _get_with_retry -------------------------------------------------------


# A handler that returns the given outcomes in order, one per request, and
# counts the requests. An exception in the list is raised instead, the way a
# real timeout would be.
def scripted(
    outcomes: list[httpx.Response | Exception],
) -> tuple[Callable[[httpx.Request], httpx.Response], list[int]]:
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = outcomes[calls[0]]
        calls[0] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return handler, calls


# Stands in for asyncio.sleep: records each delay and returns at once, so the
# tests can check the waits without spending them.
class FakeSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


async def get(
    handler: Callable[[httpx.Request], httpx.Response], sleep: FakeSleep
) -> httpx.Response:
    return await _get_with_retry(
        make_client(handler), fast_limiter(), "/v2/stocks/bars", {}, sleep=sleep
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503])
async def test_retry_recovers_from_server_error(status: int) -> None:
    handler, calls = scripted([httpx.Response(status), httpx.Response(200)])
    sleep = FakeSleep()

    response = await get(handler, sleep)

    assert response.status_code == 200
    assert calls == [2]
    assert len(sleep.delays) == 1


@pytest.mark.asyncio
async def test_retry_recovers_from_transport_error() -> None:
    handler, calls = scripted([httpx.ConnectTimeout("timed out"), httpx.Response(200)])
    sleep = FakeSleep()

    response = await get(handler, sleep)

    assert response.status_code == 200
    assert calls == [2]


@pytest.mark.asyncio
async def test_retry_429_waits_until_reset() -> None:
    reset = int(time.time()) + 10
    handler, calls = scripted(
        [
            httpx.Response(429, headers={"X-RateLimit-Reset": str(reset)}),
            httpx.Response(200),
        ]
    )
    sleep = FakeSleep()

    await get(handler, sleep)

    # About 10s; the bound allows for the test itself taking a moment, and for
    # the reset being whole seconds.
    assert calls == [2]
    [delay] = sleep.delays
    assert 8.0 < delay <= 10.0


@pytest.mark.asyncio
async def test_retry_429_without_reset_header_backs_off() -> None:
    handler, calls = scripted([httpx.Response(429), httpx.Response(200)])
    sleep = FakeSleep()

    response = await get(handler, sleep)

    assert response.status_code == 200
    assert calls == [2]
    [delay] = sleep.delays
    assert 0.0 <= delay <= alpaca.BASE_DELAY


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_retry_client_error_is_not_retried(status: int) -> None:
    # A client error means the request itself is wrong; sending it again
    # cannot succeed.
    handler, calls = scripted([httpx.Response(status)])
    sleep = FakeSleep()

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await get(handler, sleep)

    assert exc_info.value.response.status_code == status
    assert calls == [1]
    assert sleep.delays == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_retry_gives_up_after_max_attempts(status: int) -> None:
    handler, calls = scripted([httpx.Response(status)] * MAX_ATTEMPTS)
    sleep = FakeSleep()

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await get(handler, sleep)

    assert exc_info.value.response.status_code == status
    assert calls == [MAX_ATTEMPTS]
    # A wait between attempts, none after the last.
    assert len(sleep.delays) == MAX_ATTEMPTS - 1


@pytest.mark.asyncio
async def test_retry_reraises_transport_error_after_max_attempts() -> None:
    handler, calls = scripted([httpx.ConnectTimeout("timed out")] * MAX_ATTEMPTS)
    sleep = FakeSleep()

    with pytest.raises(httpx.ConnectTimeout):
        await get(handler, sleep)

    assert calls == [MAX_ATTEMPTS]
    assert len(sleep.delays) == MAX_ATTEMPTS - 1


# random.uniform is replaced with "return the upper bound", which pins down the
# ceiling each attempt's delay is drawn from.
def test_backoff_delay_doubles_up_to_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca.random, "uniform", lambda low, high: high)

    assert [_backoff_delay(a) for a in range(4)] == [1.0, 2.0, 4.0, 8.0]
    assert _backoff_delay(10) == MAX_DELAY
