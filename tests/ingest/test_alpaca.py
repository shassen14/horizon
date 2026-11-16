import datetime
from collections.abc import Callable
from decimal import Decimal

import httpx
import pytest
from pydantic import ValidationError

from horizon.ingest.alpaca import fetch_daily_bars

START = datetime.date(2026, 11, 2)
END = datetime.date(2026, 11, 6)


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

    result = await fetch_daily_bars(client, ["AAPL"], START, END)
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

    result = await fetch_daily_bars(client, ["AAPL"], START, END)
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
        await fetch_daily_bars(client, ["AAPL"], START, END)

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
        await fetch_daily_bars(make_client(handler), ["AAPL"], START, END)


@pytest.mark.asyncio
async def test_fetch_sends_raw_sip_daily_params() -> None:
    seen: list[httpx.QueryParams] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params)
        return httpx.Response(200, json={"bars": {}, "next_page_token": None})

    await fetch_daily_bars(make_client(handler), ["AAPL", "MSFT"], START, END)

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

    result = await fetch_daily_bars(make_client(handler), ["AAPL", "MSFT"], START, END)

    # The second request carried the first page's token, then the loop stopped.
    assert tokens == [None, "page-2"]
    assert [b.close for b in result["AAPL"]] == [Decimal("1.0")]
    assert [b.close for b in result["MSFT"]] == [Decimal("2.0"), Decimal("3.0")]
