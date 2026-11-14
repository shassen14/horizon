import datetime
from decimal import Decimal

import pytest
from psycopg_pool import AsyncConnectionPool

from horizon.db.assets import active_assets, upsert_assets
from horizon.db.corporate_actions import insert_actions
from horizon.db.types import (
    ActionKind,
    AssetId,
    CorporateAction,
    Dividend,
    NewAsset,
    Source,
    Split,
)

EX_DATE = datetime.date(2024, 2, 9)


async def create_asset(db: AsyncConnectionPool, symbol: str) -> AssetId:
    # Actions reference assets(id), so each test needs a real asset row first.
    await upsert_assets(
        db, [NewAsset(symbol=symbol, name=None, exchange=None, asset_type="stock")]
    )
    return next(a.id for a in await active_assets(db) if a.symbol == symbol)


def make_action(asset_id: AssetId, kind: ActionKind) -> CorporateAction:
    return CorporateAction(
        asset_id=asset_id, ex_date=EX_DATE, source=Source.ALPACA, action_kind=kind
    )


@pytest.mark.asyncio
async def test_insert_actions_empty_is_noop(db: AsyncConnectionPool) -> None:
    assert await insert_actions(db, []) == 0


@pytest.mark.asyncio
async def test_insert_actions_flattens_each_kind(db: AsyncConnectionPool) -> None:
    asset_id = await create_asset(db, "AAPL")
    # A split and a dividend on the same ex_date are distinct rows, because
    # action_type is part of the primary key.
    actions = [
        make_action(asset_id, Split(ratio=Decimal(4))),
        make_action(asset_id, Dividend(cash=Decimal("0.24"))),
    ]

    assert await insert_actions(db, actions) == 2

    # Each kind fills only its own payload column and leaves the other NULL.
    async with db.connection() as conn:
        cur = await conn.execute(
            """
            SELECT action_type, split_ratio, cash_amount
            FROM corporate_actions
            ORDER BY action_type
            """
        )
        assert await cur.fetchall() == [
            ("dividend", None, Decimal("0.24")),
            ("split", Decimal(4), None),
        ]


@pytest.mark.asyncio
async def test_insert_actions_is_idempotent(db: AsyncConnectionPool) -> None:
    asset_id = await create_asset(db, "AAPL")
    actions = [
        make_action(asset_id, Split(ratio=Decimal(4))),
        make_action(asset_id, Dividend(cash=Decimal("0.24"))),
    ]

    assert await insert_actions(db, actions) == 2
    # A recorded action never changes, so re-inserting it must skip, not fail.
    assert await insert_actions(db, actions) == 0
