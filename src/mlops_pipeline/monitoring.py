"""CLI entry point for production drift monitoring."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .data import load_dataset


def _load_production_data(prod_path: Path) -> pd.DataFrame | None:
    if not prod_path.exists():
        print(f"[monitoring] no production log found at {prod_path}; skipping drift check")
        return None
    try:
        current = pd.read_csv(prod_path)
    except Exception as e:
        print(f"[monitoring] failed to read production log: {e}")
        return None
    if len(current) < 10:
        print(f"[monitoring] too few production samples ({len(current)}) to check drift")
        return None
    return current


def _load_baseline_reference(cfg: PipelineConfig) -> pd.DataFrame | None:
    try:
        bundle = load_dataset(cfg.data, cfg.seed)
    except Exception as e:
        print(f"[monitoring] failed to load training baseline: {e}")
        return None
    return pd.DataFrame(bundle.X_train, columns=bundle.feature_names)


def _run_drift_analysis(
    reference: pd.DataFrame, current: pd.DataFrame, out_path: Path
) -> tuple[float, dict[str, float]]:
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    print(
        f"[monitoring] analyzing drift: baseline={len(reference)} rows, "
        f"production={len(current)} rows"
    )
    report = Report(metrics=[DataSummaryPreset(), DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out_path))
    print(f"[monitoring] saved drift report to {out_path}")

    metrics = result.dict()["metrics"]
    drift_metric = next(
        m
        for m in metrics
        if m.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount"
    )
    drift_share = float(drift_metric["value"]["share"])

    column_p_values = {}
    for m in metrics:
        if m.get("config", {}).get("type") == "evidently:metric_v2:ValueDrift":
            col = m["config"].get("column")
            if col is not None:
                column_p_values[col] = float(m["value"])

    return drift_share, column_p_values


def _print_drift_summary(
    drift_share: float,
    column_p_values: dict[str, float],
    per_column_drift: dict[str, float],
) -> None:
    print(f"[monitoring] drifted feature share: {drift_share:.2%}")
    for col, p_val in column_p_values.items():
        threshold = per_column_drift.get(col, 0.05)
        drift_status = "DRIFTED" if p_val < threshold else "stable"
        print(f"  - {col}: p-value={p_val:.5f} (threshold={threshold}) [{drift_status}]")


def _check_drift_gates(
    drift_share: float,
    max_drifted_share: float,
    column_p_values: dict[str, float],
    per_column_drift: dict[str, float],
) -> list[str]:
    breaches = []
    if drift_share >= max_drifted_share:
        breaches.append(f"drifted feature share {drift_share:.2%} >= limit {max_drifted_share:.2%}")
    for col, threshold in per_column_drift.items():
        if col in column_p_values and column_p_values[col] < threshold:
            breaches.append(
                f"column '{col}' drift p-value {column_p_values[col]:.5e} < threshold {threshold}"
            )
    return breaches


def monitor_drift(
    config_path: str | Path,
    production_data_path: str | Path | None = None,
    report_path: str | Path | None = None,
    fail_on_drift: bool = False,
) -> int:
    """Compare production logs against training baseline for feature drift."""
    cfg = PipelineConfig.from_yaml(config_path) if Path(config_path).exists() else PipelineConfig()

    prod_path = Path(production_data_path or cfg.serving.production_data_path)
    out_path = Path(report_path or "artifacts/reports/production_drift_report.html")

    current = _load_production_data(prod_path)
    if current is None:
        return 0

    reference = _load_baseline_reference(cfg)
    if reference is None:
        return 1

    missing = [col for col in reference.columns if col not in current.columns]
    if missing:
        print(f"[monitoring] logged production data is missing baseline columns: {missing}")
        return 1

    current = current[reference.columns]
    drift_share, column_p_values = _run_drift_analysis(reference, current, out_path)

    _print_drift_summary(drift_share, column_p_values, cfg.gates.per_column_drift)
    breaches = _check_drift_gates(
        drift_share, cfg.gates.max_drifted_share, column_p_values, cfg.gates.per_column_drift
    )

    if breaches:
        print("[monitoring] DRIFT GATES BREACHED:")
        for b in breaches:
            print(f"  - {b}")
        if fail_on_drift:
            return 1

    return 0


def main() -> None:
    """CLI wrapper for monitor_drift."""
    parser = argparse.ArgumentParser(description="Real-time production drift monitor")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--production-data", default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    exit_code = monitor_drift(
        config_path=args.config,
        production_data_path=args.production_data,
        report_path=args.report_path,
        fail_on_drift=args.fail_on_drift,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
