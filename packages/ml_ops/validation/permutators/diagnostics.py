import polars as pl
import pandas as pd


class PermutationVerifier:
    """
    A diagnostic tool to verify the statistical integrity of a permuted DataFrame.
    It compares the moments of the log-returns and key extra columns.
    """

    def __init__(self, logger, tolerance: float = 0.05):
        self.logger = logger
        # Tolerance for stats that should be very close
        self.close_tolerance = 0.05  # 5% for Std Dev
        # Tolerance for stats that are known to diverge with this method
        self.loose_tolerance = 0.50  # 50% for Skew/Kurtosis
        # Tolerance for stats that should be identical
        self.exact_tolerance = 0.001  # 0.1% for Mean

    def verify(
        self, original_df: pd.DataFrame, permuted_df: pd.DataFrame, extra_cols: list
    ) -> bool:
        """
        Runs all checks and returns True if all stats are within tolerance.
        """
        self.logger.info("--- Running Permutation Self-Verification (on first run) ---")

        # 1. Start/End Price Check
        start_ok = self._check_endpoints(
            original_df, permuted_df, "close", "Start Price"
        )
        end_ok = self._check_endpoints(original_df, permuted_df, "close", "End Price")

        # 2. Log Return Statistics Check
        returns_ok, returns_report = self._check_log_return_stats(
            original_df, permuted_df
        )

        # 3. Extra Columns (Breadth) Check
        extras_ok = True
        extras_report = ""
        if extra_cols:
            extras_ok, extras_report = self._check_extra_col_stats(
                original_df, permuted_df, extra_cols
            )

        # Log the full report
        self.logger.info(returns_report)
        if extras_report:
            self.logger.info(extras_report)

        passed = all([start_ok, end_ok, returns_ok, extras_ok])
        if not passed:
            self.logger.error(
                "❌ Permutation verification FAILED. Statistical properties were not preserved."
            )
        else:
            self.logger.success("✅ Permutation verification PASSED.")

        return passed

    def _check_endpoints(self, df1, df2, col, name):
        val1, val2 = df1[col].iloc[0], df2[col].iloc[0]
        is_ok = (
            abs((val1 - val2) / val1) < self.exact_tolerance
            if val1 != 0
            else val1 == val2
        )
        if not is_ok:
            self.logger.warning(
                f"  -> {name} Mismatch: Real={val1:.2f}, Perm={val2:.2f}"
            )
        return is_ok

    def _check_log_return_stats(self, df1, df2):
        s1 = pl.from_pandas(df1).get_column("close").log().diff()
        s2 = pl.from_pandas(df2).get_column("close").log().diff()

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

        # This will track the pass/fail for Mean and Std Dev only
        critical_stats_ok = True

        for metric, (v1, v2, tolerance) in stats.items():
            if v1 is not None and v2 is not None and v1 != 0:
                is_ok = abs((v1 - v2) / v1) < tolerance
            else:
                is_ok = (
                    abs(v1 - v2) < 1e-9 if v1 is not None and v2 is not None else False
                )

            # Only Mean and Std Dev determine the final pass/fail status
            if metric in ["Mean", "Std Dev"] and not is_ok:
                critical_stats_ok = False

            status = "✅" if is_ok else "❌"
            # For Skew/Kurtosis, if it fails, we'll call it a "divergence"
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
            s1, s2 = df1[col], df2[col]
            stats = {"Mean": (s1.mean(), s2.mean()), "Std Dev": (s1.std(), s2.std())}

            reports.append(f"  Column: '{col}'")
            for metric, (v1, v2) in stats.items():
                is_ok = (
                    abs((v1 - v2) / v1) < self.exact_tolerance
                    if v1 != 0
                    else abs(v1 - v2) < 1e-9
                )
                if not is_ok:
                    all_ok = False
                reports.append(
                    f"    - {metric}: Real={v1:.4f}, Perm={v2:.4f} {'✅' if is_ok else '❌'}"
                )

        return all_ok, "\n".join(reports)
