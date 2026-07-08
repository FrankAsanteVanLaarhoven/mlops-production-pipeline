import pandas as pd

from mlops_pipeline.adapters.ray_serving import ModelService
from mlops_pipeline.monitoring import monitor_drift
from mlops_pipeline.registry import register_model


def test_monitor_drift_missing_log(tmp_path):
    # If production data doesn't exist, it should return 0 (skipped)
    res = monitor_drift(
        config_path="configs/pipeline.yaml",
        production_data_path=tmp_path / "non_existent.csv",
        report_path=tmp_path / "report.html",
    )
    assert res == 0


def test_monitor_drift_too_few_samples(tmp_path):
    # Less than 10 samples should skip and return 0
    df = pd.DataFrame([[0.5] * 10], columns=[f"feature_{i}" for i in range(10)])
    log_path = tmp_path / "prod.csv"
    df.to_csv(log_path, index=False)

    res = monitor_drift(
        config_path="configs/pipeline.yaml",
        production_data_path=log_path,
        report_path=tmp_path / "report.html",
    )
    assert res == 0


def test_monitor_drift_runs_successfully(tmp_path):
    # Create 15 production samples (above the 10 threshold)
    features = [f"feature_{i}" for i in range(10)]
    data = [[0.1] * 10] * 15
    df = pd.DataFrame(data, columns=features)
    log_path = tmp_path / "prod.csv"
    df.to_csv(log_path, index=False)

    report_path = tmp_path / "report.html"
    res = monitor_drift(
        config_path="configs/pipeline.yaml",
        production_data_path=log_path,
        report_path=report_path,
        fail_on_drift=False,
    )
    assert res == 0
    assert report_path.exists()


def test_serving_logs_predictions(tmp_path, trained, bundle):
    model, params = trained
    # Register the model to a temp registry root
    registry_root = tmp_path / "registry"
    register_model(
        model=model,
        architecture={
            "n_features": bundle.n_features,
            "hidden_dim": params["hidden_dim"],
            "weight_bit_width": params["weight_bit_width"],
        },
        params=params,
        metrics={"accuracy": 0.95},
        drift_share=0.0,
        data_fingerprint=bundle.fingerprint(),
        root=registry_root,
        dvc_track=False,
        feature_names=bundle.feature_names,
    )

    # Initialize ModelService with the temp registry and a production data path
    prod_data_path = tmp_path / "production_data.csv"
    service = ModelService.func_or_class(
        registry_root=str(registry_root),
        max_abs_feature_value=10.0,
        production_data_path=str(prod_data_path),
    )

    # Call log prediction directly
    service._log_prediction([0.5] * 10)
    assert prod_data_path.exists()

    df = pd.read_csv(prod_data_path)
    assert len(df) == 1
    assert list(df.columns) == bundle.feature_names
    assert [float(x) for x in list(df.iloc[0])] == [0.5] * 10


def test_monitor_drift_read_csv_exception(tmp_path):
    # Pass a directory path which will raise an IsADirectoryError / Exception in read_csv
    res = monitor_drift(
        config_path="configs/pipeline.yaml",
        production_data_path=tmp_path,
        report_path=tmp_path / "report.html",
    )
    assert res == 0


def test_monitor_drift_load_baseline_reference_failure(tmp_path):
    # Set csv_path to a non-existent file to force load_dataset failure
    import yaml
    cfg_data = {
        "data": {
            "source": "csv",
            "csv_path": str(tmp_path / "non_existent_data.csv"),
        }
    }
    config_file = tmp_path / "bad_pipeline.yaml"
    config_file.write_text(yaml.dump(cfg_data))

    # We need production data to exist and have >= 10 samples so it proceeds to baseline load
    features = [f"feature_{i}" for i in range(10)]
    data = [[0.1] * 10] * 15
    df = pd.DataFrame(data, columns=features)
    log_path = tmp_path / "prod.csv"
    df.to_csv(log_path, index=False)

    res = monitor_drift(
        config_path=config_file,
        production_data_path=log_path,
        report_path=tmp_path / "report.html",
    )
    assert res == 1


def test_monitor_drift_missing_columns(tmp_path):
    # Production columns don't match reference columns
    df = pd.DataFrame([[0.5] * 5] * 12, columns=[f"feature_{i}" for i in range(5)])
    log_path = tmp_path / "prod_bad_cols.csv"
    df.to_csv(log_path, index=False)

    res = monitor_drift(
        config_path="configs/pipeline.yaml",
        production_data_path=log_path,
        report_path=tmp_path / "report.html",
    )
    assert res == 1


def test_monitor_drift_fail_on_drift_gate_breached(tmp_path):
    # Setup production data and a config that enforces drift check and column drift threshold
    import yaml
    cfg_data = {
        "gates": {
            "max_drifted_share": 0.0,  # force global breach
            "per_column_drift": {"feature_0": 0.99},  # force per-column breach
        }
    }
    config_file = tmp_path / "drift_pipeline.yaml"
    config_file.write_text(yaml.dump(cfg_data))

    features = [f"feature_{i}" for i in range(10)]
    data = [[0.1] * 10] * 15
    df = pd.DataFrame(data, columns=features)
    log_path = tmp_path / "prod.csv"
    df.to_csv(log_path, index=False)

    # With fail_on_drift=True, should return 1
    res = monitor_drift(
        config_path=config_file,
        production_data_path=log_path,
        report_path=tmp_path / "report.html",
        fail_on_drift=True,
    )
    assert res == 1


def test_monitoring_main(tmp_path, monkeypatch):
    import sys
    import mlops_pipeline.monitoring as monitoring_module

    exit_codes = []
    monkeypatch.setattr(sys, "exit", lambda code: exit_codes.append(code))
    
    # We point to a non-existent CSV to skip gracefully with 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlops-monitor",
            "--config",
            "configs/pipeline.yaml",
            "--production-data",
            str(tmp_path / "non_existent.csv"),
        ],
    )
    monitoring_module.main()
    assert exit_codes == [0]

