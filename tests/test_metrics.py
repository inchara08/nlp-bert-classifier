from src.evaluation.metrics import compute_metrics, aggregate_fold_metrics


def test_compute_metrics_perfect():
    y = [0, 1, 2, 0, 1, 2]
    metrics = compute_metrics(y, y)
    assert metrics["weighted_f1"] == 1.0
    assert metrics["weighted_precision"] == 1.0
    assert metrics["weighted_recall"] == 1.0


def test_compute_metrics_keys():
    metrics = compute_metrics([0, 1, 2], [0, 1, 0])
    assert "weighted_f1" in metrics
    assert "macro_f1" in metrics
    assert "weighted_precision" in metrics
    assert "weighted_recall" in metrics


def test_aggregate_fold_metrics_mean():
    folds = [
        {"weighted_f1": 0.88, "macro_f1": 0.85},
        {"weighted_f1": 0.90, "macro_f1": 0.87},
    ]
    agg = aggregate_fold_metrics(folds)
    assert abs(agg["weighted_f1"]["mean"] - 0.89) < 1e-9


def test_aggregate_fold_metrics_std():
    folds = [
        {"weighted_f1": 0.89, "macro_f1": 0.86},
        {"weighted_f1": 0.89, "macro_f1": 0.86},
    ]
    agg = aggregate_fold_metrics(folds)
    assert agg["weighted_f1"]["std"] == 0.0


def test_aggregate_has_std_key():
    folds = [{"weighted_f1": 0.88}, {"weighted_f1": 0.90}]
    agg = aggregate_fold_metrics(folds)
    assert "std" in agg["weighted_f1"]
    assert "mean" in agg["weighted_f1"]
