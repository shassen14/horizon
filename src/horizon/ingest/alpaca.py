import asyncio
import datetime
import json
import random
import time
from collections.abc import Awaitable, Callable
from decimal import Decimal

import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel, ConfigDict, Field

from horizon.config import alpaca_keys

# Retry policy for one request: up to 5 tries, with backoff delays (in seconds)
# that double from BASE_DELAY and never exceed MAX_DELAY.
MAX_ATTEMPTS = 5
BASE_DELAY = 1.0
MAX_DELAY = 30.0


# One bar as /v2/stocks/bars returns it. Aliases map Alpaca's one-letter keys to
# readable names, and validate_by_name lets tests build bars by those names.
# Keys not listed here, such as n (trade count), are ignored.
class AlpacaBar(BaseModel):
    model_config = ConfigDict(frozen=True, validate_by_name=True)

    open: Decimal = Field(alias="o")
    high: Decimal = Field(alias="h")
    low: Decimal = Field(alias="l")
    close: Decimal = Field(alias="c")
    volume: int = Field(alias="v")
    vwap: Decimal | None = Field(default=None, alias="vw")
    # Midnight New York time, expressed in UTC (05:00Z in winter, 04:00Z in
    # summer). DailyKey.from_et_instant turns it into the trading date.
    time: datetime.datetime = Field(alias="t")


# One page of a response; next_page_token is None on the last page.
class BarsPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    bars: dict[str, list[AlpacaBar]]
    next_page_token: str | None


# The settings every market-data request shares: host, auth headers, timeout.
# Make one per run and use it as `async with make_client() as client:`, which
# closes it on every exit path and reuses its connections across batches. The
# timeout is above httpx's 5s default because a full 10,000-bar page can be slow.
def make_client() -> httpx.AsyncClient:
    alpaca_key, alpaca_secret = alpaca_keys()
    headers: dict[str, str] = {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret,
    }

    return httpx.AsyncClient(
        base_url="https://data.alpaca.markets", headers=headers, timeout=30.0
    )


# Alpaca's free plan allows 200 requests a minute; 190 leaves headroom. Make one
# per run and share it across all concurrent batches: the limit is per account,
# and a limiter only sees the requests that go through it.
def make_limiter() -> AsyncLimiter:
    return AsyncLimiter(190, 60)


# Daily only for now; timeframe becomes a parameter once a second timeframe has a
# caller. Both start and end dates are inclusive.
async def fetch_daily_bars(
    client: httpx.AsyncClient,
    limiter: AsyncLimiter,
    symbols: list[str],
    start: datetime.date,
    end: datetime.date,
) -> dict[str, list[AlpacaBar]]:
    params: dict[str, str | int] = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        # Unadjusted prices; splits and dividends are stored separately.
        "adjustment": "raw",
        # All US exchanges. The free plan delays it by 15 minutes, which a daily
        # job never notices; iex alone carries only a few percent of volume.
        "feed": "sip",
        # The maximum, counted across all symbols combined, not per symbol.
        "limit": 10_000,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    result: dict[str, list[AlpacaBar]] = {}

    while True:
        response = await _get_with_retry(client, limiter, "/v2/stocks/bars", params)

        # parse_float=Decimal keeps prices exact; as floats they would already
        # have lost precision before pydantic saw them.
        page = BarsPage.model_validate(json.loads(response.text, parse_float=Decimal))

        # A symbol's bars can continue on the next page, so extend, never replace.
        for symbol, bars in page.bars.items():
            result.setdefault(symbol, []).extend(bars)

        if page.next_page_token is None:
            break

        params["page_token"] = page.next_page_token

    return result


# Sends one GET, retrying what can succeed on a second try:
#   429           wait until X-RateLimit-Reset, or back off if it is missing
#   5xx, timeouts back off, since the server or network is briefly unwell
#   other 4xx     raise at once; the request itself is wrong, so resending
#                 it cannot help
# Each attempt passes through the limiter, so retries count toward the rate
# limit. The last failure is raised rather than returned, so callers only
# ever see a successful response. sleep is a parameter so tests can record the
# waits without spending them.
async def _get_with_retry(
    client: httpx.AsyncClient,
    limiter: AsyncLimiter,
    path: str,
    params: dict[str, str | int],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> httpx.Response:
    for attempt in range(MAX_ATTEMPTS):
        try:
            async with limiter:
                response = await client.get(path, params=params)
        except httpx.TransportError:
            if attempt == MAX_ATTEMPTS - 1:
                raise

            await sleep(_backoff_delay(attempt))
            continue

        status = response.status_code

        if status < 400:
            return response

        if status == 429:
            reset = response.headers.get("X-RateLimit-Reset")
            if reset is None:
                delay = _backoff_delay(attempt)
            else:
                delay = max(0.0, int(reset) - time.time())

            if attempt == MAX_ATTEMPTS - 1:
                response.raise_for_status()

            await sleep(delay)
            continue
        elif status >= 500:
            if attempt == MAX_ATTEMPTS - 1:
                response.raise_for_status()
            await sleep(_backoff_delay(attempt))
            continue
        else:
            response.raise_for_status()

    # Every branch above returns or raises on the last attempt; this line only
    # satisfies the type checker, which can't see that the loop always runs.
    raise AssertionError("unreachable")


# "Full jitter": a random delay between 0 and a ceiling that doubles each
# attempt. The randomness spreads out batches that failed together, so they
# don't all retry at the same instant and overload the server again.
def _backoff_delay(attempt: int) -> float:
    return random.uniform(0, min(MAX_DELAY, BASE_DELAY * 2**attempt))
