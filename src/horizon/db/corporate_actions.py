from decimal import Decimal

from psycopg_pool import AsyncConnectionPool

from horizon.db.types import CorporateAction, Dividend, Split


async def insert_actions(
    pool: AsyncConnectionPool, actions: list[CorporateAction]
) -> int:
    if not actions:
        return 0

    # known
    asset_ids = [action.asset_id for action in actions]
    ex_dates = [action.ex_date for action in actions]
    sources = [action.source.value for action in actions]

    # derived
    action_types: list[str] = []
    split_ratios: list[Decimal | None] = []
    cash_amounts: list[Decimal | None] = []

    for action in actions:
        match action.action_kind:
            case Split(ratio=ratio):
                action_types.append("split")
                split_ratios.append(ratio)
                cash_amounts.append(None)
            case Dividend(cash=cash):
                action_types.append("dividend")
                split_ratios.append(None)
                cash_amounts.append(cash)

    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO corporate_actions
                (asset_id, ex_date, action_type, split_ratio, cash_amount, source)
            SELECT asset_id, ex_date, action_type, split_ratio, cash_amount, source
            FROM UNNEST(
                %s::int4[], %s::date[], %s::text[],
                %s::numeric[], %s::numeric[], %s::text[]
            ) AS t(asset_id, ex_date, action_type, split_ratio, cash_amount, source)
            ON CONFLICT (asset_id, ex_date, action_type) DO NOTHING
            """,
            (asset_ids, ex_dates, action_types, split_ratios, cash_amounts, sources),
        )

    return cur.rowcount
