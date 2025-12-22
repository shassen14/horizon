"""convert_to_hypertables

Revision ID: 85bf3a4a553a
Revises: d222678603db
Create Date: 2025-12-20 11:40:44.661547

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "85bf3a4a553a"
down_revision: Union[str, Sequence[str], None] = "d222678603db"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# NOTE: Manually wrote this to convert to hypertables for effeciency
def upgrade() -> None:
    # 1. Market Data Daily (1 Year chunks)
    op.execute(
        """
        SELECT create_hypertable(
            'market_data_daily', 'time', 
            chunk_time_interval => INTERVAL '1 year', 
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    """
    )

    # 2. Market Data 5Min (1 Day chunks - Critical for performance)
    op.execute(
        """
        SELECT create_hypertable(
            'market_data_5min', 'time', 
            chunk_time_interval => INTERVAL '1 day', 
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    """
    )

    # 3. Features (1 Year chunks)
    op.execute(
        """
        SELECT create_hypertable(
            'features_daily', 'time', 
            chunk_time_interval => INTERVAL '1 year', 
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
    """
    )


def downgrade() -> None:
    # We generally do not revert hypertable conversion because it's a destructive storage change.
    # To revert, you usually have to drop the table.
    pass
