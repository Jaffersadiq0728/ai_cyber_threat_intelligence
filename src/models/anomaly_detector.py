"""
models/anomaly_detector.py
--------------------------
Two-stage threat detection:

  Stage 1 — Unsupervised anomaly scoring
    IsolationForest assigns an anomaly score to every flow.
    Flows with score > threshold are flagged as suspicious.

  Stage 2 — Supervised attack classification
    Random Forest multi-class classifier trained on labelled data.
    Predicts the specific attack type for suspicious flows.

Both models are trained from scratch on the CIC-IDS-2017 feature set.
No pre-trained weights, no API calls.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)


# ─── Isolation Forest ─────────────────────────────────────────────────────────

class AnomalyScorer:
    """
    Wraps IsolationForest.
    Trained ONLY on benign traffic so it learns what 'normal' looks like.
    Outputs a [0,1] anomaly score for each flow.
    """

    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.model = None

    def fit(self, X_benign):
        """Train on benign-only traffic."""
        print(f"  [AnomalyScorer] Fitting on {len(X_benign):,} benign samples...")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_benign)
        print("  [AnomalyScorer] Done.")
        return self

    def score(self, X):
        """
        Return anomaly scores in [0, 1].
        Higher score = more anomalous.
        IsolationForest returns negative_outlier_factor; we invert & normalise.
        """
        if self.model is None:
            raise RuntimeError("AnomalyScorer not fitted yet.")
        raw = self.model.score_samples(X)          # lower = more anomalous
        # Shift to [0,1]: 0 = normal, 1 = highly anomalous
        lo, hi = raw.min(), raw.max()
        if hi == lo:
            return np.zeros(len(X))
        normalised = 1.0 - (raw - lo) / (hi - lo)
        return normalised.astype(np.float32)

    def predict(self, X, threshold=0.60):
        """Return binary labels: 1=anomaly, 0=normal."""
        return (self.score(X) >= threshold).astype(int)


# ─── Random Forest Classifier ─────────────────────────────────────────────────

class AttackClassifier:
    """
    Multi-class classifier that maps network-flow features to attack types.
    Trained on the full labelled CIC-IDS-2017 dataset.
    """

    def __init__(self, n_estimators=150, max_depth=20, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self.model        = None
        self.label_encoder = None
        self.feature_importances_ = None

    def fit(self, X_train, y_train, label_encoder=None):
        print(f"  [AttackClassifier] Training on {len(X_train):,} samples...")
        self.label_encoder = label_encoder
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        print("  [AttackClassifier] Done.")
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("AttackClassifier not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("AttackClassifier not fitted yet.")
        return self.model.predict_proba(X)

    def predict_label(self, X):
        """Return human-readable class names."""
        y_pred = self.predict(X)
        if self.label_encoder:
            return self.label_encoder.inverse_transform(y_pred)
        return y_pred.astype(str)

    def confidence(self, X):
        """Return max-probability confidence for each sample."""
        proba = self.predict_proba(X)
        return proba.max(axis=1)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        print(f"\n  [AttackClassifier] Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
        if self.label_encoder:
            target_names = list(self.label_encoder.classes_)
        else:
            target_names = None
        print(classification_report(y_test, y_pred,
                                    target_names=target_names,
                                    zero_division=0))
        return {"accuracy": acc, "f1_weighted": f1,
                "y_pred": y_pred, "y_test": y_test}

    def top_features(self, feature_names, top_n=15):
        if self.feature_importances_ is None:
            return []
        idx = np.argsort(self.feature_importances_)[::-1][:top_n]
        return [(feature_names[i], round(float(self.feature_importances_[i]), 4))
                for i in idx]


# ─── Combined Detector ────────────────────────────────────────────────────────

class ThreatDetector:
    """
    Combines AnomalyScorer and AttackClassifier.
    Pipeline:
      1. Score every flow with IsolationForest
      2. For flows scoring >= threshold, classify with RandomForest
      3. Return enriched result dicts
    """

    def __init__(self, anomaly_threshold=0.60):
        self.anomaly_threshold = anomaly_threshold
        self.scorer     = AnomalyScorer()
        self.classifier = AttackClassifier()
        self.fitted     = False

    def fit(self, data):
        """
        data: dict from preprocessing_pipeline()
        """
        X_train    = data["X_train"]
        y_train    = data["y_train"]
        le         = data["label_encoder"]

        # Train anomaly scorer on benign flows only
        benign_class = list(le.classes_).index("BENIGN") if "BENIGN" in le.classes_ else None
        if benign_class is not None:
            benign_mask = (y_train == benign_class)
            X_benign = X_train[benign_mask]
        else:
            X_benign = X_train

        self.scorer.fit(X_benign)
        self.classifier.fit(X_train, y_train, label_encoder=le)
        self.fitted = True
        return self

    def analyse(self, X, feature_names=None, raw_rows=None):
        """
        Analyse a batch of flows.

        Returns list of dicts:
          {
            "index":        int,
            "anomaly_score": float,        # 0-1, higher = more suspicious
            "is_anomaly":    bool,
            "attack_type":   str,
            "confidence":    float,
            "threat_score":  float,        # combined final score
          }
        """
        if not self.fitted:
            raise RuntimeError("ThreatDetector.fit() must be called first.")

        anomaly_scores = self.scorer.score(X)
        attack_labels  = self.classifier.predict_label(X)
        confidences    = self.classifier.confidence(X)

        results = []
        for i in range(len(X)):
            a_score    = float(anomaly_scores[i])
            attack     = str(attack_labels[i])
            conf       = float(confidences[i])
            is_anomaly = (a_score >= self.anomaly_threshold)

            # Threat score: blend anomaly score + classifier confidence for attacks
            if attack == "BENIGN":
                threat_score = a_score * 0.4
            else:
                threat_score = min(1.0, a_score * 0.5 + conf * 0.5)

            result = {
                "index":         i,
                "anomaly_score": round(a_score, 4),
                "is_anomaly":    is_anomaly,
                "attack_type":   attack,
                "confidence":    round(conf, 4),
                "threat_score":  round(threat_score, 4),
            }

            if raw_rows is not None:
                row = raw_rows[i] if isinstance(raw_rows, list) else raw_rows.iloc[i]
                result["source_ip"]  = str(row.get("Source IP", ""))
                result["dest_ip"]    = str(row.get("Destination IP", ""))
                result["timestamp"]  = str(row.get("Timestamp", ""))
                result["src_port"]   = int(row.get("Source Port", 0) or 0)
                result["dst_port"]   = int(row.get("Destination Port", 0) or 0)

            results.append(result)

        return results

    def evaluate(self, data):
        return self.classifier.evaluate(data["X_test"], data["y_test"])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  [ThreatDetector] Saved to {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
