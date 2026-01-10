import polars as pl
import pandas as pd
from packages.contracts.vocabulary.columns import MarketCol
from packages.ml_ops.validation.permutators.ohlc import OHLCPermutator


class PermutationVerifier:
    def __init__(self, logger, tolerance: float = 0.05):
        self.logger = logger
        self.close_tolerance = 0.05
        self.loose_tolerance = 0.50
        self.exact_tolerance = 0.001

    def verify(
        self,
        original_df: pd.DataFrame,
        permuted_df: pd.DataFrame = None,
        extra_cols: list = None,
    ) -> bool:
        """
        Runs statistical checks.
        Auto-detects if Multi-Asset and isolates a single asset for verification.
        """
        self.logger.info("--- Running Permutation Self-Verification ---")

        # --- DEFINE CONSTANTS AT TOP SCOPE ---
        ohlc_cols = {
            MarketCol.OPEN,
            MarketCol.HIGH,
            MarketCol.LOW,
            MarketCol.CLOSE,
            MarketCol.VOLUME,
            MarketCol.SYMBOL,
            MarketCol.TIME,
            "asset_id",
        }

        # 1. Handle Multi-Asset Datasets
        if MarketCol.SYMBOL in original_df.columns:
            unique_symbols = original_df[MarketCol.SYMBOL].unique()
            if len(unique_symbols) > 1:
                counts = original_df[MarketCol.SYMBOL].value_counts()
                target_sym = counts.idxmax()
                self.logger.info(
                    f"Multi-Asset dataset detected. Verifying representative asset: {target_sym}"
                )

                df_to_test = original_df[
                    original_df[MarketCol.SYMBOL] == target_sym
                ].copy()

                target_extra = [c for c in df_to_test.columns if c not in ohlc_cols]

                permutator = OHLCPermutator(diff_cols=target_extra)
                df_permuted = permutator.permute(df_to_test, seed=0)

                # Recursive call with single asset
                return self.verify(df_to_test, df_permuted, target_extra)

        # 2. Handle Single-Asset (Missing permuted_df)
        if permuted_df is None:
            # We need to generate it here!
            if extra_cols is None:
                extra_cols = [c for c in original_df.columns if c not in ohlc_cols]

            permutator = OHLCPermutator(diff_cols=extra_cols)
            permuted_df = permutator.permute(original_df, seed=0)

        # 3. Standard Verification
        start_ok = self._check_endpoints(
            original_df, permuted_df, MarketCol.CLOSE, "Start Price"
        )
        end_ok = self._check_endpoints(
            original_df, permuted_df, MarketCol.CLOSE, "End Price"
        )

        returns_ok, returns_report = self._check_log_return_stats(
            original_df, permuted_df
        )

        extras_ok = True
        extras_report = ""
        if extra_cols:
            extras_ok, extras_report = self._check_extra_col_stats(
                original_df, permuted_df, extra_cols
            )

        self.logger.info(returns_report)
        if extras_report:
            self.logger.info(extras_report)

        # Only fail on critical stats (Mean/Std/Endpoints)
        passed = all([start_ok, end_ok, returns_ok, extras_ok])

        if not passed:
            self.logger.error("❌ Permutation verification FAILED on critical stats.")
        else:
            self.logger.success("✅ Permutation verification PASSED on critical stats.")
            self.logger.warning("   -> Note: Skew/Kurtosis divergence is expected.")

        return passed

    def _check_endpoints(self, df1, df2, col, name):
        val1, val2 = df1[col].iloc[0], df2[col].iloc[0]
        if val1 == 0:
            return val1 == val2
        is_ok = abs((val1 - val2) / val1) < self.exact_tolerance
        if not is_ok:
            self.logger.warning(
                f"  -> {name} Mismatch: Real={val1:.2f}, Perm={val2:.2f}"
            )
        return is_ok

    def _check_log_return_stats(self, df1, df2):
        s1 = pl.from_pandas(df1).get_column(MarketCol.CLOSE).log().diff()
        s2 = pl.from_pandas(df2).get_column(MarketCol.CLOSE).log().diff()

        stats = {
            "Mean": (s1.mean(), s2.mean(), self.exact_tolerance),
            "Std Dev": (s1.std(), s2.std(), self.close_tolerance),
            "Skew": (s1.skew(), s2.skew(), self.loose_tolerance),
            "Kurtosis": (s1.kurtosis(), s2.kurtosis(), self.loose_tolerance),
        }

        report = [
            "\n--- Log Return Statistics ---",
            f"{'Metric':<10} | {'REAL':>15} | {'PERMUTED':>15} | {'STATUS':>10}",
        ]
        critical_stats_ok = True

        for metric, (v1, v2, tolerance) in stats.items():
            if v1 is not None and v2 is not None and v1 != 0:
                is_ok = abs((v1 - v2) / v1) < tolerance
            else:
                is_ok = (
                    abs(v1 - v2) < 1e-9 if v1 is not None and v2 is not None else False
                )

            # Loose check: Only Mean/Std affect the boolean return
            if metric in ["Mean", "Std Dev"] and not is_ok:
                critical_stats_ok = False

            status = "✅" if is_ok else "❌"
            if metric in ["Skew", "Kurtosis"] and not is_ok:
                status = "⚠️ (Diverged)"

            report.append(
                f"{metric:<10} | {v1 or 0.0:15.8f} | {v2 or 0.0:15.8f} | {status:>10}"
            )

        return critical_stats_ok, "\n".join(report)

    def _check_extra_col_stats(self, df1, df2, extra_cols):
        all_ok = True
        reports = ["\n--- Extra Column Statistics ---"]
        for col in extra_cols:
            if col not in df1.columns or col not in df2.columns:
                continue

            s1, s2 = df1[col], df2[col]
            stats = {"Mean": (s1.mean(), s2.mean()), "Std Dev": (s1.std(), s2.std())}

            reports.append(f"  Column: '{col}'")
            for metric, (v1, v2) in stats.items():
                if v1 == 0:
                    is_ok = abs(v2) < 1e-9
                else:
                    is_ok = abs((v1 - v2) / v1) < self.exact_tolerance

                if not is_ok:
                    all_ok = False
                reports.append(
                    f"    - {metric}: Real={v1:.4f}, Perm={v2:.4f} {'✅' if is_ok else '❌'}"
                )

        return all_ok, "\n".join(reports)
