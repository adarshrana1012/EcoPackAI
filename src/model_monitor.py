"""
model_monitor.py
================

Real-time model monitoring for the EcoPackAI platform.

Provides a :class:`ModelMonitor` that tracks prediction confidence,
detects feature drift via KL divergence, and detects label drift via
chi-squared tests.  All anomalies are surfaced as structured
:class:`MonitoringAlert` objects.

Typical usage
-------------
>>> from src.model_monitor import ModelMonitor
>>> monitor = ModelMonitor("packaging_classifier", confidence_threshold=0.75)
>>> monitor.log_prediction(features={"weight": 3.2}, prediction=1, confidence=0.91)
>>> alerts = monitor.run_all_checks(recent_features=X_new, feature_names=names)
>>> report = monitor.get_health_report()

Author:  EcoPackAI Team
Created: 2026-06-12
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_WINDOW_SIZE: int = 100
_HISTOGRAM_BINS: int = 50
_KL_EPSILON: float = 1e-10
_CHI2_ALPHA: float = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MonitoringAlert:
    """Represents a single monitoring alert raised by :class:`ModelMonitor`.

    Attributes
    ----------
    alert_type : str
        Category of the alert.  One of ``'confidence_drift'``,
        ``'feature_drift'``, or ``'label_drift'``.
    severity : str
        Severity level – ``'warning'`` or ``'critical'``.
    message : str
        Human-readable description of the issue.
    metric_value : float
        The observed metric value that triggered the alert.
    threshold : float
        The threshold that was exceeded (or not met).
    timestamp : datetime
        UTC timestamp when the alert was created.
    """

    alert_type: str
    severity: str
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# ModelMonitor
# ---------------------------------------------------------------------------
class ModelMonitor:
    """Monitors an ML model's health in production.

    The monitor tracks three dimensions of model health:

    1. **Confidence drift** – whether mean prediction confidence has
       dropped below an acceptable threshold.
    2. **Feature drift** – whether the distribution of incoming features
       has diverged from the training (reference) distributions, measured
       via KL divergence on histograms.
    3. **Label drift** – whether the distribution of predicted labels has
       shifted relative to a reference distribution, measured via a
       chi-squared test.

    Parameters
    ----------
    model_name : str
        A human-readable identifier for the model being monitored.
    confidence_threshold : float, optional
        Alert if the mean confidence over the most recent window drops
        below this value.  Defaults to ``0.75``.
    drift_threshold : float, optional
        KL-divergence threshold for feature drift alerts.  Defaults to
        ``0.1``.
    reference_distributions : dict[str, numpy.ndarray] | None, optional
        Per-feature reference (training) distributions.  Keys are feature
        names, values are 1-D arrays of training-set values.
    window_size : int, optional
        Number of most-recent predictions to consider for windowed
        checks.  Defaults to ``100``.
    """

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.75,
        drift_threshold: float = 0.1,
        reference_distributions: Optional[Dict[str, np.ndarray]] = None,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> None:
        self.model_name: str = model_name
        self.confidence_threshold: float = confidence_threshold
        self.drift_threshold: float = drift_threshold
        self.reference_distributions: Dict[str, np.ndarray] = (
            reference_distributions if reference_distributions is not None else {}
        )
        self.window_size: int = window_size

        # Internal state
        self._alerts: List[MonitoringAlert] = []
        self._prediction_log: List[Dict[str, Any]] = []

        logger.info(
            "ModelMonitor initialised for '%s' "
            "(confidence_threshold=%.2f, drift_threshold=%.4f, window=%d).",
            self.model_name,
            self.confidence_threshold,
            self.drift_threshold,
            self.window_size,
        )

    # ------------------------------------------------------------------ #
    # Prediction logging
    # ------------------------------------------------------------------ #
    def log_prediction(
        self,
        features: Dict[str, Any],
        prediction: int,
        confidence: float,
    ) -> None:
        """Record a single prediction for downstream monitoring.

        Parameters
        ----------
        features : dict
            Feature name → value mapping used for this prediction.
        prediction : int
            The predicted class label.
        confidence : float
            Model confidence / predicted probability for the chosen class.
        """
        entry: Dict[str, Any] = {
            "features": features,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._prediction_log.append(entry)
        logger.debug(
            "Logged prediction for '%s': pred=%d, conf=%.4f.",
            self.model_name,
            prediction,
            confidence,
        )

    # ------------------------------------------------------------------ #
    # Confidence check
    # ------------------------------------------------------------------ #
    def check_confidence_distribution(self) -> Optional[MonitoringAlert]:
        """Check whether recent prediction confidence is acceptable.

        Examines the most recent ``window_size`` predictions and computes
        the mean confidence.  If the mean falls below
        ``confidence_threshold`` an alert is created.

        Returns
        -------
        MonitoringAlert or None
            An alert if mean confidence is below threshold, otherwise
            ``None``.
        """
        if not self._prediction_log:
            logger.warning(
                "No predictions logged for '%s'; skipping confidence check.",
                self.model_name,
            )
            return None

        recent = self._prediction_log[-self.window_size :]
        confidences = np.array([p["confidence"] for p in recent], dtype=np.float64)
        mean_conf: float = float(np.mean(confidences))

        logger.info(
            "Confidence check for '%s': mean=%.4f over %d predictions "
            "(threshold=%.2f).",
            self.model_name,
            mean_conf,
            len(recent),
            self.confidence_threshold,
        )

        if mean_conf < self.confidence_threshold:
            severity = "critical" if mean_conf < self.confidence_threshold * 0.8 else "warning"
            alert = MonitoringAlert(
                alert_type="confidence_drift",
                severity=severity,
                message=(
                    f"Mean prediction confidence for '{self.model_name}' dropped "
                    f"to {mean_conf:.4f} (threshold: {self.confidence_threshold:.2f}) "
                    f"over the last {len(recent)} predictions."
                ),
                metric_value=mean_conf,
                threshold=self.confidence_threshold,
            )
            self._alerts.append(alert)
            logger.warning("Confidence drift alert [%s]: %s", severity, alert.message)
            return alert

        return None

    # ------------------------------------------------------------------ #
    # Feature drift (KL divergence)
    # ------------------------------------------------------------------ #
    def check_feature_drift(
        self,
        recent_features: np.ndarray,
        feature_names: List[str],
    ) -> List[MonitoringAlert]:
        """Detect feature drift using histogram-based KL divergence.

        For each feature, the method:

        1. Builds normalised histograms of the reference and recent
           distributions using a common bin range.
        2. Adds a small epsilon (``1e-10``) to avoid ``log(0)``.
        3. Computes *D*:sub:`KL`\\ ``(recent ‖ reference)``.

        Parameters
        ----------
        recent_features : numpy.ndarray
            2-D array of shape ``(n_samples, n_features)`` with recent
            incoming data.
        feature_names : list[str]
            Names corresponding to columns in *recent_features*.

        Returns
        -------
        list[MonitoringAlert]
            One alert per feature whose KL divergence exceeds
            ``drift_threshold``.
        """
        alerts: List[MonitoringAlert] = []

        if not self.reference_distributions:
            logger.warning(
                "No reference distributions configured for '%s'; "
                "skipping feature drift check.",
                self.model_name,
            )
            return alerts

        if recent_features.ndim == 1:
            recent_features = recent_features.reshape(-1, 1)

        if recent_features.shape[1] != len(feature_names):
            logger.error(
                "Feature matrix has %d columns but %d feature names provided.",
                recent_features.shape[1],
                len(feature_names),
            )
            raise ValueError(
                f"Mismatch between feature matrix columns "
                f"({recent_features.shape[1]}) and feature_names length "
                f"({len(feature_names)})."
            )

        for idx, fname in enumerate(feature_names):
            ref_values = self.reference_distributions.get(fname)
            if ref_values is None:
                logger.debug(
                    "No reference distribution for feature '%s'; skipping.", fname
                )
                continue

            recent_col = recent_features[:, idx].astype(np.float64)
            ref_col = ref_values.astype(np.float64)

            kl_div = self._kl_divergence_histogram(recent_col, ref_col)

            logger.debug(
                "KL divergence for feature '%s': %.6f (threshold=%.4f).",
                fname,
                kl_div,
                self.drift_threshold,
            )

            if kl_div > self.drift_threshold:
                severity = "critical" if kl_div > self.drift_threshold * 2.0 else "warning"
                alert = MonitoringAlert(
                    alert_type="feature_drift",
                    severity=severity,
                    message=(
                        f"Feature '{fname}' shows drift: KL divergence = "
                        f"{kl_div:.6f} (threshold: {self.drift_threshold:.4f})."
                    ),
                    metric_value=kl_div,
                    threshold=self.drift_threshold,
                )
                alerts.append(alert)
                self._alerts.append(alert)
                logger.warning("Feature drift alert [%s]: %s", severity, alert.message)

        logger.info(
            "Feature drift check complete for '%s': %d alert(s) raised.",
            self.model_name,
            len(alerts),
        )
        return alerts

    # ------------------------------------------------------------------ #
    # Label drift (chi-squared)
    # ------------------------------------------------------------------ #
    def check_label_drift(
        self,
        recent_labels: np.ndarray,
        reference_labels: np.ndarray,
    ) -> Optional[MonitoringAlert]:
        """Detect label distribution drift using a chi-squared test.

        Both arrays are binned into class counts and compared via
        :func:`scipy.stats.chisquare`.  If the test's *p*-value falls
        below ``0.05`` the shift is considered significant.

        Parameters
        ----------
        recent_labels : numpy.ndarray
            1-D array of recent predicted labels.
        reference_labels : numpy.ndarray
            1-D array of reference (training / validation) labels.

        Returns
        -------
        MonitoringAlert or None
            An alert if the chi-squared test indicates significant drift.
        """
        recent_labels = np.asarray(recent_labels)
        reference_labels = np.asarray(reference_labels)

        # Determine the full set of classes across both arrays
        all_classes = np.union1d(np.unique(recent_labels), np.unique(reference_labels))

        recent_counts = np.array(
            [np.sum(recent_labels == c) for c in all_classes], dtype=np.float64
        )
        ref_counts = np.array(
            [np.sum(reference_labels == c) for c in all_classes], dtype=np.float64
        )

        # Normalise reference counts to the same total as recent counts so
        # that the chi-squared comparison is on the same scale.
        ref_total = ref_counts.sum()
        recent_total = recent_counts.sum()
        if ref_total == 0 or recent_total == 0:
            logger.warning(
                "Empty label array supplied; skipping label drift check."
            )
            return None

        expected = ref_counts * (recent_total / ref_total)
        # Guard against zeros in expected (would cause division error).
        expected = np.where(expected == 0, _KL_EPSILON, expected)

        chi2_stat, p_value = scipy_stats.chisquare(f_obs=recent_counts, f_exp=expected)

        logger.info(
            "Label drift check for '%s': chi2=%.4f, p=%.6f (alpha=%.2f).",
            self.model_name,
            chi2_stat,
            p_value,
            _CHI2_ALPHA,
        )

        if p_value < _CHI2_ALPHA:
            severity = "critical" if p_value < 0.01 else "warning"
            alert = MonitoringAlert(
                alert_type="label_drift",
                severity=severity,
                message=(
                    f"Label distribution drift detected for '{self.model_name}': "
                    f"chi2={chi2_stat:.4f}, p-value={p_value:.6f} "
                    f"(alpha={_CHI2_ALPHA})."
                ),
                metric_value=float(p_value),
                threshold=_CHI2_ALPHA,
            )
            self._alerts.append(alert)
            logger.warning("Label drift alert [%s]: %s", severity, alert.message)
            return alert

        return None

    # ------------------------------------------------------------------ #
    # Aggregate checks
    # ------------------------------------------------------------------ #
    def run_all_checks(
        self,
        recent_features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        recent_labels: Optional[np.ndarray] = None,
        reference_labels: Optional[np.ndarray] = None,
    ) -> List[MonitoringAlert]:
        """Execute all available monitoring checks and return alerts.

        Each check is run only if the required data has been supplied.
        The confidence check always runs if predictions have been logged.

        Parameters
        ----------
        recent_features : numpy.ndarray or None
            2-D feature matrix for drift detection.
        feature_names : list[str] or None
            Column names matching *recent_features*.
        recent_labels : numpy.ndarray or None
            Recent predicted labels for label-drift detection.
        reference_labels : numpy.ndarray or None
            Reference labels to compare against.

        Returns
        -------
        list[MonitoringAlert]
            Aggregated list of alerts from all executed checks.
        """
        logger.info("Running all monitoring checks for '%s'.", self.model_name)
        all_alerts: List[MonitoringAlert] = []

        # 1. Confidence
        conf_alert = self.check_confidence_distribution()
        if conf_alert is not None:
            all_alerts.append(conf_alert)

        # 2. Feature drift
        if recent_features is not None and feature_names is not None:
            feature_alerts = self.check_feature_drift(recent_features, feature_names)
            all_alerts.extend(feature_alerts)
        else:
            logger.debug(
                "Skipping feature drift check – missing recent_features or "
                "feature_names."
            )

        # 3. Label drift
        if recent_labels is not None and reference_labels is not None:
            label_alert = self.check_label_drift(recent_labels, reference_labels)
            if label_alert is not None:
                all_alerts.append(label_alert)
        else:
            logger.debug(
                "Skipping label drift check – missing recent_labels or "
                "reference_labels."
            )

        logger.info(
            "All checks complete for '%s': %d total alert(s).",
            self.model_name,
            len(all_alerts),
        )
        return all_alerts

    # ------------------------------------------------------------------ #
    # Alert management
    # ------------------------------------------------------------------ #
    def get_alerts(self, severity: Optional[str] = None) -> List[MonitoringAlert]:
        """Retrieve stored alerts, optionally filtered by severity.

        Parameters
        ----------
        severity : str or None
            If provided, only alerts matching this severity level
            (``'warning'`` or ``'critical'``) are returned.

        Returns
        -------
        list[MonitoringAlert]
        """
        if severity is not None:
            filtered = [a for a in self._alerts if a.severity == severity]
            logger.debug(
                "Returning %d alert(s) with severity='%s'.",
                len(filtered),
                severity,
            )
            return filtered
        return list(self._alerts)

    def clear_alerts(self) -> None:
        """Remove all stored alerts."""
        count = len(self._alerts)
        self._alerts.clear()
        logger.info("Cleared %d alert(s) for '%s'.", count, self.model_name)

    # ------------------------------------------------------------------ #
    # Health report
    # ------------------------------------------------------------------ #
    def get_health_report(self) -> Dict[str, Any]:
        """Generate a summary health report for the monitored model.

        Returns
        -------
        dict
            Keys include:

            * ``model_name`` – model identifier.
            * ``total_predictions`` – number of logged predictions.
            * ``mean_confidence`` – mean confidence over the window.
            * ``min_confidence`` – minimum observed confidence in window.
            * ``max_confidence`` – maximum observed confidence in window.
            * ``confidence_threshold`` – configured threshold.
            * ``drift_threshold`` – configured KL divergence threshold.
            * ``total_alerts`` – total number of alerts raised.
            * ``critical_alerts`` – count of critical-severity alerts.
            * ``warning_alerts`` – count of warning-severity alerts.
            * ``alert_breakdown`` – count per ``alert_type``.
            * ``status`` – ``'healthy'``, ``'degraded'``, or ``'unhealthy'``.
            * ``generated_at`` – ISO-formatted UTC timestamp.
        """
        total = len(self._prediction_log)
        window = self._prediction_log[-self.window_size :] if total > 0 else []
        confidences = np.array(
            [p["confidence"] for p in window], dtype=np.float64
        ) if window else np.array([], dtype=np.float64)

        critical_count = sum(1 for a in self._alerts if a.severity == "critical")
        warning_count = sum(1 for a in self._alerts if a.severity == "warning")

        # Build alert-type breakdown
        alert_breakdown: Dict[str, int] = {}
        for alert in self._alerts:
            alert_breakdown[alert.alert_type] = (
                alert_breakdown.get(alert.alert_type, 0) + 1
            )

        # Determine overall status
        if critical_count > 0:
            status = "unhealthy"
        elif warning_count > 0:
            status = "degraded"
        else:
            status = "healthy"

        report: Dict[str, Any] = {
            "model_name": self.model_name,
            "total_predictions": total,
            "mean_confidence": float(np.mean(confidences)) if len(confidences) > 0 else None,
            "min_confidence": float(np.min(confidences)) if len(confidences) > 0 else None,
            "max_confidence": float(np.max(confidences)) if len(confidences) > 0 else None,
            "confidence_threshold": self.confidence_threshold,
            "drift_threshold": self.drift_threshold,
            "total_alerts": len(self._alerts),
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "alert_breakdown": alert_breakdown,
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Health report for '%s': status=%s, alerts=%d (critical=%d, warning=%d).",
            self.model_name,
            status,
            len(self._alerts),
            critical_count,
            warning_count,
        )
        return report

    # ================================================================== #
    # Private helpers
    # ================================================================== #
    @staticmethod
    def _kl_divergence_histogram(
        p_samples: np.ndarray,
        q_samples: np.ndarray,
        bins: int = _HISTOGRAM_BINS,
        epsilon: float = _KL_EPSILON,
    ) -> float:
        """Compute KL divergence D_KL(P ‖ Q) using histogram approximation.

        Both sample arrays are discretised into a shared set of bins.  A
        small *epsilon* is added to every bin count to avoid ``log(0)``.

        Parameters
        ----------
        p_samples : numpy.ndarray
            Samples drawn from distribution P (recent / observed).
        q_samples : numpy.ndarray
            Samples drawn from distribution Q (reference / training).
        bins : int, optional
            Number of histogram bins.  Defaults to ``50``.
        epsilon : float, optional
            Small constant added to bin counts.  Defaults to ``1e-10``.

        Returns
        -------
        float
            Non-negative KL divergence value.
        """
        # Determine shared bin edges from the union of both ranges.
        combined_min = min(float(np.min(p_samples)), float(np.min(q_samples)))
        combined_max = max(float(np.max(p_samples)), float(np.max(q_samples)))

        # Protect against degenerate case (all values identical).
        if combined_min == combined_max:
            return 0.0

        bin_edges = np.linspace(combined_min, combined_max, bins + 1)

        p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=False)
        q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=False)

        # Normalise to probability distributions and add epsilon.
        p_dist = (p_hist.astype(np.float64) + epsilon)
        p_dist /= p_dist.sum()

        q_dist = (q_hist.astype(np.float64) + epsilon)
        q_dist /= q_dist.sum()

        kl: float = float(np.sum(p_dist * np.log(p_dist / q_dist)))
        return max(kl, 0.0)  # clip numerical noise
