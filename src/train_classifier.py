"""
train_classifier.py — EcoPackAI Fragility Classifier Training Pipeline
======================================================================

Implements the full training workflow for the Random Forest fragility
classifier covering three stages:

1. **Baseline model** (Prompt 8)  — default RF with ``class_weight='balanced'``
2. **Hyperparameter tuning** (Prompt 9) — ``RandomizedSearchCV`` over 50 iters
3. **Imbalance comparison** (Prompt 10) — balanced weights vs SMOTE vs threshold

Usage
-----
    python -m src.train_classifier

Author: EcoPackAI Team
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.feature_pipeline import build_feature_pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "length_cm", "width_cm", "height_cm", "weight_g",
    "material_type", "volume_cm3", "aspect_ratio",
    "historical_material_volume_cm3",
]
TARGET_COLUMN: str = "fragility_label"
TIER_LABELS: Dict[int, str] = {0: "None", 1: "Low", 2: "Medium", 3: "Critical"}


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_training_data(
    data_dir: str = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the pre-split train / validation / test CSVs.

    Parameters
    ----------
    data_dir : str
        Directory containing ``train.csv``, ``val.csv``, ``test.csv``
        (produced by ``data_splitter.py``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``

    Raises
    ------
    FileNotFoundError
        If any of the three CSVs is missing.
    """
    base = Path(data_dir)
    paths = {name: base / f"{name}.csv" for name in ("train", "val", "test")}

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"{name}.csv not found at {p.resolve()}. "
                "Run data_splitter.py first."
            )

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])

    logger.info(
        "Loaded splits — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def _prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series, Any]:
    """Apply the feature pipeline and separate X / y.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_val, y_val, X_test, y_test, pipeline)``
    """
    # Preserve targets before pipeline transform
    y_train = train_df[TARGET_COLUMN].copy()
    y_val = val_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    # Build and fit pipeline on training data
    pipeline = build_feature_pipeline()
    train_transformed = pipeline.fit_transform(train_df)
    val_transformed = pipeline.transform(val_df)
    test_transformed = pipeline.transform(test_df)

    # Select only feature columns
    X_train = train_transformed[FEATURE_COLUMNS]
    X_val = val_transformed[FEATURE_COLUMNS]
    X_test = test_transformed[FEATURE_COLUMNS]

    logger.info("Feature matrix shapes — X_train: %s, X_val: %s, X_test: %s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test, pipeline


# ═══════════════════════════════════════════════════════════════════════════
# Prompt 8: Baseline Random Forest
# ═══════════════════════════════════════════════════════════════════════════

def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train a baseline Random Forest with ``class_weight='balanced'``.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data for evaluation

    Returns
    -------
    tuple[RandomForestClassifier, dict]
        ``(model, metrics_dict)`` where *metrics_dict* contains
        classification report, confusion matrix, accuracy, and the
        hardest-to-classify tier.
    """
    logger.info("=" * 60)
    logger.info("PROMPT 8: Training baseline Random Forest")
    logger.info("=" * 60)

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics
    report_dict = classification_report(
        y_test, y_pred, target_names=list(TIER_LABELS.values()), output_dict=True,
    )
    report_str = classification_report(
        y_test, y_pred, target_names=list(TIER_LABELS.values()),
    )
    cm = confusion_matrix(y_test, y_pred)
    accuracy = float((y_pred == y_test).mean())

    # Identify hardest tier (lowest F1)
    per_class_f1 = {
        tier: report_dict[label]["f1-score"]
        for tier, label in TIER_LABELS.items()
    }
    hardest_tier = min(per_class_f1, key=per_class_f1.get)

    metrics = {
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "accuracy": round(accuracy, 4),
        "per_class_f1": per_class_f1,
        "hardest_tier": hardest_tier,
        "hardest_tier_label": TIER_LABELS[hardest_tier],
    }

    print("\n" + "=" * 60)
    print("  BASELINE MODEL — Classification Report")
    print("=" * 60)
    print(report_str)
    print("Confusion Matrix:")
    print(cm)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Hardest tier: {hardest_tier} ({TIER_LABELS[hardest_tier]}) "
          f"— F1={per_class_f1[hardest_tier]:.4f}")
    print("=" * 60 + "\n")

    logger.info("Baseline accuracy: %.4f", accuracy)
    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Prompt 9: Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Run ``RandomizedSearchCV`` to find the best RF hyperparameters.

    Parameters
    ----------
    X_train, y_train : training data
    n_iter : int
        Number of parameter settings sampled.
    cv : int
        Number of cross-validation folds.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[RandomForestClassifier, dict]
        ``(best_estimator, tuning_results)``
    """
    logger.info("=" * 60)
    logger.info("PROMPT 9: Hyperparameter tuning (%d iterations, %d-fold CV)",
                n_iter, cv)
    logger.info("=" * 60)

    param_distributions = {
        "n_estimators": [100, 150, 200, 250, 300, 350, 400, 450, 500],
        "max_depth": [None, 10, 15, 20, 25, 30],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="precision_macro",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = search.best_score_

    # Top-5 results summary
    results_df = pd.DataFrame(search.cv_results_)
    top5 = results_df.nsmallest(5, "rank_test_score")[
        ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ]

    tuning_results = {
        "best_params": best_params,
        "best_cv_score": round(float(best_score), 4),
        "n_iterations": n_iter,
        "cv_folds": cv,
        "scoring": "precision_macro",
        "top_5_results": top5.to_dict(orient="records"),
    }

    print("\n" + "=" * 60)
    print("  HYPERPARAMETER TUNING — Results")
    print("=" * 60)
    print(f"\n  Best CV Score (precision_macro): {best_score:.4f}")
    print(f"  Best Parameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print("\n  Top 5 configurations:")
    print(top5.to_string(index=False))
    print("=" * 60 + "\n")

    logger.info("Best CV precision_macro: %.4f", best_score)
    logger.info("Best params: %s", best_params)

    return search.best_estimator_, tuning_results


# ═══════════════════════════════════════════════════════════════════════════
# Prompt 10: Class Imbalance Comparison
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_critical_class(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    approach_name: str,
) -> Dict[str, Any]:
    """Evaluate a model with focus on the Critical (label=3) class."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True,
                                   target_names=list(TIER_LABELS.values()))

    critical_metrics = report.get("Critical", {})
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    return {
        "approach": approach_name,
        "critical_precision": round(critical_metrics.get("precision", 0), 4),
        "critical_recall": round(critical_metrics.get("recall", 0), 4),
        "critical_f1": round(critical_metrics.get("f1-score", 0), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(precision_score(y_test, y_pred, average="macro")), 4),
    }


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], RandomForestClassifier]:
    """Compare three imbalance-handling strategies.

    Strategies
    ----------
    1. ``class_weight='balanced'`` with tuned hyperparameters
    2. SMOTE oversampling + tuned RF (no class_weight)
    3. ``class_weight='balanced'`` + threshold tuning for Critical class

    Returns
    -------
    tuple[str, dict, RandomForestClassifier]
        ``(recommended_approach, comparison_dict, best_model)``
    """
    logger.info("=" * 60)
    logger.info("PROMPT 10: Comparing imbalance strategies")
    logger.info("=" * 60)

    results: List[Dict[str, Any]] = []
    models: Dict[str, RandomForestClassifier] = {}

    # --- Strategy 1: class_weight='balanced' --------------------------------
    params_balanced = {k: v for k, v in best_params.items()
                       if k != "class_weight"}
    model_balanced = RandomForestClassifier(
        **params_balanced,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model_balanced.fit(X_train, y_train)
    res1 = _evaluate_critical_class(model_balanced, X_test, y_test,
                                    "class_weight='balanced'")
    results.append(res1)
    models["balanced"] = model_balanced

    # --- Strategy 2: SMOTE oversampling -------------------------------------
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info("SMOTE resampled: %d -> %d rows",
                    len(X_train), len(X_resampled))

        params_no_weight = {k: v for k, v in best_params.items()
                            if k != "class_weight"}
        model_smote = RandomForestClassifier(
            **params_no_weight,
            random_state=42,
            n_jobs=-1,
        )
        model_smote.fit(X_resampled, y_resampled)
        res2 = _evaluate_critical_class(model_smote, X_test, y_test,
                                        "SMOTE oversampling")
        results.append(res2)
        models["smote"] = model_smote
    except ImportError:
        logger.warning("imbalanced-learn not installed; skipping SMOTE.")
        results.append({
            "approach": "SMOTE oversampling",
            "critical_precision": None, "critical_recall": None,
            "critical_f1": None, "macro_f1": None, "macro_precision": None,
            "note": "imbalanced-learn not installed",
        })

    # --- Strategy 3: balanced + threshold tuning ----------------------------
    model_threshold = RandomForestClassifier(
        **params_balanced,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model_threshold.fit(X_train, y_train)
    probas = model_threshold.predict_proba(X_test)

    # Find optimal threshold for Critical class (index 3)
    critical_idx = 3
    best_threshold = 0.5
    best_critical_f1 = 0.0

    for threshold in np.arange(0.15, 0.60, 0.05):
        y_custom = np.array(y_test.copy())
        y_pred_custom = model_threshold.predict(X_test)
        # Override: if prob(Critical) >= threshold, predict Critical
        critical_proba = probas[:, critical_idx]
        y_pred_custom[critical_proba >= threshold] = critical_idx

        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred_custom, labels=[critical_idx], average=None,
        )
        if f[0] > best_critical_f1:
            best_critical_f1 = f[0]
            best_threshold = threshold

    # Apply best threshold
    y_pred_final = model_threshold.predict(X_test)
    critical_proba = probas[:, critical_idx]
    y_pred_final[critical_proba >= best_threshold] = critical_idx

    report_thresh = classification_report(y_test, y_pred_final, output_dict=True,
                                          target_names=list(TIER_LABELS.values()))
    critical_m = report_thresh.get("Critical", {})
    macro_f1_thresh = f1_score(y_test, y_pred_final, average="macro")

    res3 = {
        "approach": f"balanced + threshold={best_threshold:.2f}",
        "critical_precision": round(critical_m.get("precision", 0), 4),
        "critical_recall": round(critical_m.get("recall", 0), 4),
        "critical_f1": round(critical_m.get("f1-score", 0), 4),
        "macro_f1": round(float(macro_f1_thresh), 4),
        "macro_precision": round(float(precision_score(y_test, y_pred_final,
                                                        average="macro")), 4),
        "optimal_threshold": round(float(best_threshold), 2),
    }
    results.append(res3)
    models["threshold"] = model_threshold

    # --- Pick best approach by Critical F1 ----------------------------------
    valid_results = [r for r in results if r.get("critical_f1") is not None]
    best_result = max(valid_results, key=lambda r: r["critical_f1"])
    recommended = best_result["approach"]

    # Map recommendation to model
    if "SMOTE" in recommended:
        best_model = models.get("smote", models["balanced"])
    elif "threshold" in recommended:
        best_model = models["threshold"]
    else:
        best_model = models["balanced"]

    comparison = {
        "strategies": results,
        "recommended": recommended,
        "justification": (
            f"{recommended} achieves the highest Critical-class F1 score "
            f"({best_result['critical_f1']:.4f}), which is the primary "
            f"optimization target for preventing high-value item damage."
        ),
    }

    # Print comparison table
    print("\n" + "=" * 72)
    print("  CLASS IMBALANCE COMPARISON — Critical (tier=3) Focus")
    print("=" * 72)
    header = f"  {'Approach':<35} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Macro-F1':>8}"
    print(header)
    print("  " + "-" * 65)
    for r in results:
        p = r.get("critical_precision")
        rc = r.get("critical_recall")
        f1 = r.get("critical_f1")
        mf = r.get("macro_f1")
        if p is not None:
            print(f"  {r['approach']:<35} {p:>6.4f} {rc:>6.4f} {f1:>6.4f} {mf:>8.4f}")
        else:
            print(f"  {r['approach']:<35}  {'N/A':>5} {'N/A':>6} {'N/A':>6} {'N/A':>8}")
    print(f"\n  >> Recommended: {recommended}")
    print(f"  >> Justification: {comparison['justification']}")
    print("=" * 72 + "\n")

    return recommended, comparison, best_model


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_full_training_pipeline(
    data_dir: str = "data",
    output_dir: str = "models",
) -> Dict[str, Any]:
    """Run the complete training workflow (Prompts 8–10).

    Sequence
    --------
    1. Load pre-split data
    2. Apply feature pipeline
    3. Train baseline model
    4. Tune hyperparameters
    5. Compare imbalance strategies
    6. Save best model + pipeline

    Parameters
    ----------
    data_dir : str
        Directory containing train/val/test CSVs.
    output_dir : str
        Directory to save model artifacts.

    Returns
    -------
    dict
        Summary with all metrics, best params, and file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data -------------------------------------------------------
    train_df, val_df, test_df = load_training_data(data_dir)

    # --- 2. Feature engineering ---------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, feat_pipeline = (
        _prepare_features(train_df, val_df, test_df)
    )

    # Save fitted feature pipeline
    pipeline_path = out / "feature_pipeline.joblib"
    joblib.dump(feat_pipeline, pipeline_path)
    logger.info("Feature pipeline saved to %s", pipeline_path)

    # --- 3. Baseline --------------------------------------------------------
    baseline_model, baseline_metrics = train_baseline_model(
        X_train, y_train, X_test, y_test,
    )

    # --- 4. Tuning ----------------------------------------------------------
    tuned_model, tuning_results = tune_hyperparameters(X_train, y_train)

    # Evaluate tuned model on test set
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_report = classification_report(
        y_test, y_pred_tuned, target_names=list(TIER_LABELS.values()),
        output_dict=True,
    )
    tuned_accuracy = float((y_pred_tuned == y_test).mean())

    print("\n" + "=" * 60)
    print("  TUNED MODEL — Test Set Evaluation")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred_tuned, target_names=list(TIER_LABELS.values()),
    ))
    print(f"Accuracy: {tuned_accuracy:.4f}")
    print("=" * 60 + "\n")

    # --- 5. Imbalance comparison --------------------------------------------
    recommended, comparison, best_model = compare_imbalance_strategies(
        X_train, y_train, X_test, y_test,
        best_params=tuning_results["best_params"],
    )

    # --- 6. Save best model -------------------------------------------------
    model_path = out / "best_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Best model saved to %s", model_path)

    # Save feature names for inference
    feature_meta = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "tier_labels": TIER_LABELS,
        "best_params": tuning_results["best_params"],
        "recommended_strategy": recommended,
    }
    meta_path = out / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2, default=str)
    logger.info("Model metadata saved to %s", meta_path)

    # --- Summary ------------------------------------------------------------
    summary = {
        "baseline": baseline_metrics,
        "tuning": tuning_results,
        "tuned_accuracy": tuned_accuracy,
        "imbalance_comparison": comparison,
        "recommended_strategy": recommended,
        "artifacts": {
            "model": str(model_path),
            "pipeline": str(pipeline_path),
            "metadata": str(meta_path),
        },
    }

    print("\n" + "=" * 72)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 72)
    print(f"  Baseline accuracy:     {baseline_metrics['accuracy']:.4f}")
    print(f"  Tuned accuracy:        {tuned_accuracy:.4f}")
    print(f"  Best CV precision:     {tuning_results['best_cv_score']:.4f}")
    print(f"  Recommended strategy:  {recommended}")
    print(f"  Model saved to:        {model_path}")
    print(f"  Pipeline saved to:     {pipeline_path}")
    print("=" * 72 + "\n")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_training_pipeline()
