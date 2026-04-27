"""
engine/threat_engine.py
------------------------
Orchestrates the full AI Cyber Threat Intelligence pipeline:

  1. Preprocess CIC-IDS-2017 data
  2. Train AnomalyScorer + AttackClassifier
  3. Build NLP log analyser
  4. Run inference on the dataset
  5. Build and score attack graph
  6. Produce final alert objects ready for the dashboard
"""

import os
import sys
import json
import time
import uuid
import pickle
import numpy as np
import pandas as pd

# ── local imports ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from preprocessing.cleaner  import preprocess_pipeline
from models.anomaly_detector import ThreatDetector
from nlp.log_analyser        import LogNLPAnalyser
from graph.network_graph     import AttackGraph


# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_CACHE    = os.path.join(BASE, "..", "output", "threat_detector.pkl")
NLP_CACHE      = os.path.join(BASE, "..", "output", "nlp_analyser.pkl")
RESULTS_PATH   = os.path.join(BASE, "..", "output", "results.json")
REPORT_PATH    = os.path.join(BASE, "..", "reports", "threat_report.json")
DATA_DIR       = os.path.join(BASE, "..", "data")


# ─── Main Engine ──────────────────────────────────────────────────────────────

class ThreatEngine:

    def __init__(self, force_retrain=False):
        self.force_retrain = force_retrain
        self.detector   = None
        self.nlp        = None
        self.graph      = AttackGraph()
        self.alerts     = []
        self.model_metrics = {}
        self.graph_data    = {}
        self.anomaly_series = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        """Load data, train models, cache artefacts."""
        print("\n[ENGINE] === TRAINING PHASE ===")
        t0 = time.time()

        # Preprocessing
        data = preprocess_pipeline(DATA_DIR)
        self._data = data

        # ML model
        print("\n[ENGINE] Training ThreatDetector...")
        self.detector = ThreatDetector(anomaly_threshold=0.60)
        self.detector.fit(data)

        # Evaluation
        print("\n[ENGINE] Evaluating classifier...")
        metrics = self.detector.evaluate(data)
        self.model_metrics = {
            "accuracy":    round(float(metrics["accuracy"]), 4),
            "f1_weighted": round(float(metrics["f1_weighted"]), 4),
        }

        # NLP analyser
        print("\n[ENGINE] Building NLP analyser...")
        self.nlp = LogNLPAnalyser(similarity_threshold=0.25)
        self.nlp.build_reference()

        # Cache
        os.makedirs(os.path.dirname(MODEL_CACHE), exist_ok=True)
        self.detector.save(MODEL_CACHE)
        with open(NLP_CACHE, "wb") as f:
            pickle.dump(self.nlp, f)

        elapsed = time.time() - t0
        print(f"\n[ENGINE] Training complete in {elapsed:.1f}s")
        return self

    def load_models(self):
        """Load cached models if available and not force_retrain."""
        if (not self.force_retrain and
                os.path.exists(MODEL_CACHE) and os.path.exists(NLP_CACHE)):
            print("[ENGINE] Loading cached models...")
            self.detector = ThreatDetector.load(MODEL_CACHE)
            with open(NLP_CACHE, "rb") as f:
                self.nlp = pickle.load(f)
            return True
        return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def run_inference(self, max_rows=2000):
        """
        Run full inference pipeline on the loaded dataset.
        Produces self.alerts, self.graph_data, self.anomaly_series.
        """
        print("\n[ENGINE] === INFERENCE PHASE ===")
        if self.detector is None or self.nlp is None:
            raise RuntimeError("Models not loaded. Call train() or load_models() first.")

        data  = self._data
        df    = data["clean_df"]
        scaler = data["scaler"]
        le     = data["label_encoder"]
        feat_names = data["feature_names"]

        # Take a sample for inference
        sample_df = df.sample(min(max_rows, len(df)), random_state=99).reset_index(drop=True)
        print(f"  Running inference on {len(sample_df):,} flows...")

        # Feature matrix
        available = [c for c in feat_names if c in sample_df.columns]
        X_sample  = scaler.transform(
            sample_df[available].values.astype(np.float32))

        # Convert to raw dicts for NLP
        raw_rows = sample_df.to_dict(orient="records")

        # ML detection
        detections = self.detector.analyse(X_sample, feat_names, raw_rows=sample_df)

        # NLP enrichment
        print("  NLP enrichment...")
        enriched = self.nlp.enrich_detections(detections, raw_rows)

        # Build alerts (only threats above low threshold)
        self.alerts = []
        for i, e in enumerate(enriched):
            if e["threat_score"] < 0.25 and e["attack_type"] == "BENIGN":
                continue
            alert = {
                "id":            str(uuid.uuid4())[:8],
                "source_ip":     e.get("source_ip",  "0.0.0.0"),
                "dest_ip":       e.get("dest_ip",    "0.0.0.0"),
                "attack_type":   e["attack_type"],
                "threat_score":  e["threat_score"],
                "severity":      e["severity"],
                "anomaly_score": e["anomaly_score"],
                "nlp_similarity":e["nlp_similarity"],
                "timestamp":     e.get("timestamp", ""),
                "explanation":   e["explanation"],
                "mitre":         e["mitre"],
                "log_text":      e["log_text"],
            }
            self.alerts.append(alert)

            # Feed to graph
            self.graph.add_flow({
                "source_ip":   e.get("source_ip", ""),
                "dest_ip":     e.get("dest_ip", ""),
                "attack_type": e["attack_type"],
                "threat_score":e["threat_score"],
            })

        # Sort alerts by threat score
        self.alerts.sort(key=lambda x: -x["threat_score"])

        # Anomaly time series (last 60 points)
        self.anomaly_series = [
            round(float(e["anomaly_score"]), 4) for e in enriched[:60]
        ]

        # Graph export
        self.graph_data = self.graph.to_d3_format(max_nodes=50)

        print(f"  Alerts generated: {len(self.alerts)}")
        print(f"  Graph: {self.graph_data['nodes'].__len__()} nodes, "
              f"{self.graph_data['links'].__len__()} edges")
        return self

    # ── MITRE summary ─────────────────────────────────────────────────────────

    def mitre_summary(self):
        """Aggregate MITRE ATT&CK tactics from alerts."""
        tally = {}
        for a in self.alerts:
            tactic = (a["mitre"] or {}).get("tactic")
            if tactic:
                if tactic not in tally:
                    tally[tactic] = {"count": 0, "techniques": set()}
                tally[tactic]["count"] += 1
                tid = (a["mitre"] or {}).get("technique_id")
                if tid:
                    tally[tactic]["techniques"].add(tid)
        # Serialise sets
        for k in tally:
            tally[k]["techniques"] = list(tally[k]["techniques"])
        return tally

    # ── Report ────────────────────────────────────────────────────────────────

    def build_report(self):
        """Compile full JSON report."""
        g_stats = self.graph.summary_stats()
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_metrics": self.model_metrics,
            "summary": {
                "total_alerts":   len(self.alerts),
                "critical":       sum(1 for a in self.alerts if a["severity"] == "critical"),
                "high":           sum(1 for a in self.alerts if a["severity"] == "high"),
                "medium":         sum(1 for a in self.alerts if a["severity"] == "medium"),
                "low":            sum(1 for a in self.alerts if a["severity"] == "low"),
                "graph_nodes":    g_stats["total_nodes"],
                "graph_edges":    g_stats["total_edges"],
            },
            "top_attackers":  self.graph.top_attackers(10),
            "mitre_summary":  self.mitre_summary(),
            "alerts":         self.alerts[:100],
            "graph":          self.graph_data,
            "anomaly_series": self.anomaly_series,
        }

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[ENGINE] Report saved → {REPORT_PATH}")

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(report, f, default=str)
        print(f"[ENGINE] Results saved → {RESULTS_PATH}")

        return report

    # ── Full pipeline shortcut ────────────────────────────────────────────────

    def run(self, max_rows=2000):
        loaded = self.load_models()
        if not loaded:
            self.train()
        # Preprocessing needed for inference even if models cached
        if not hasattr(self, "_data"):
            print("\n[ENGINE] Loading data for inference...")
            self._data = preprocess_pipeline(DATA_DIR)
        self.run_inference(max_rows=max_rows)
        return self.build_report()
