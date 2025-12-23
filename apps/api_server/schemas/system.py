from pydantic import BaseModel
from datetime import datetime
from .enums import SystemHealth, Environment


class SystemStatus(BaseModel):
    status: SystemHealth
    last_daily_update: datetime | None
    last_intraday_update: datetime | None
    active_assets: int
    environment: Environment
