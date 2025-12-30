from abc import ABC, abstractmethod
import polars as pl
from typing import Dict, Any


class AbstractBacktester(ABC):
    @abstractmethod
    def run(self, df: pl.DataFrame, target_col: str, pred_col: str) -> Dict[str, Any]:
        """
        Executes the backtest simulation.
        df must contain: time, asset_id, target_col, pred_col
        """
        pass
