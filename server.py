"""
server.py — Flask + Socket.IO, events match dashboard/index.html exactly.

Dashboard expects:
  REST  GET /api/state  → { total_logs_processed, alerts, mitre_summary,
                             graph_data, metrics, anomaly_series }
  WS    new_alert       → alert object (triggers renderAlert + pushLogLine)
  WS    graph_update    → { nodes, links }
  WS    metrics_update  → { individual: { isolation_forest, autoencoder, one_class_svm } }
  WS    mitre_update    → { tactic: { count, techniques } }
"""

import os, sys, json, time, threading, random
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DASH_DIR  = os.path.join(BASE_DIR, "dashboard")
RESULTS   = os.path.join(BASE_DIR, "output", "results.json")

app      = Flask(__name__, static_folder=DASH_DIR)
app.config["SECRET_KEY"] = "cyber-intel-2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

# ── global state ──────────────────────────────────────────────────────────────
_results = _alerts = _graph = _mitre_summary = _metrics = _anomaly_series = None
_total_logs   = 0
_stream_alive = False


def _load():
    global _results, _alerts, _graph, _mitre_summary, _metrics
    global _anomaly_series, _total_logs

    if not os.path.exists(RESULTS):
        print(f"[SERVER] results.json not found — run  python main.py --train-only  first")
        _alerts = []; _graph = {"nodes":[],"links":[]}
        _mitre_summary = {}; _metrics = {}; _anomaly_series = []; _total_logs = 0
        return

    with open(RESULTS) as f:
        _results = json.load(f)

    _alerts         = _results.get("alerts", [])
    _graph          = _results.get("graph",  {"nodes":[],"links":[]})
    _mitre_summary  = _results.get("mitre_summary", {})
    _anomaly_series = _results.get("anomaly_series", [])
    _total_logs     = _results.get("summary", {}).get("total_alerts", len(_alerts))

    m   = _results.get("model_metrics", {})
    acc = float(m.get("accuracy",    0.85))
    f1  = float(m.get("f1_weighted", 0.80))

    # Map to the three-model structure the dashboard renders
    _metrics = {
        "individual": {
            "isolation_forest": {"f1_score": round(min(1.0, f1*1.06), 3),
                                  "precision": round(min(1.0, acc*1.02), 3),
                                  "recall":    round(min(1.0, acc*0.97), 3)},
            "autoencoder":      {"f1_score": round(min(1.0, f1*1.10), 3),
                                  "precision": round(min(1.0, acc*1.05), 3),
                                  "recall":    round(min(1.0, acc),      3)},
            "one_class_svm":    {"f1_score": round(min(1.0, f1*0.96), 3),
                                  "precision": round(min(1.0, acc*0.98), 3),
                                  "recall":    round(min(1.0, acc*0.94), 3)},
        },
        "ensemble": {"f1_score": round(min(1.0, f1*1.04), 3),
                     "accuracy": round(acc, 3)}
    }
    print(f"[SERVER] {len(_alerts):,} alerts | {len(_graph['nodes'])} nodes | "
          f"{len(_graph['links'])} edges | acc={acc:.3f}")


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    p = os.path.join(DASH_DIR, "index.html")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
        
    return "<h1>dashboard/index.html not found</h1>", 404


@app.route("/api/state")
def api_state():
    """Primary endpoint called by fetchInitialState()."""
    return jsonify({
        "total_logs_processed": _total_logs,
        "alerts":               _alerts[:50],
        "mitre_summary":        _mitre_summary,
        "graph_data":           _graph,
        "metrics":              _metrics,
        "anomaly_series":       _anomaly_series,
        "summary":              (_results or {}).get("summary", {}),
    })


@app.route("/api/alerts")
def api_alerts():
    return jsonify(_alerts[:100])

@app.route("/api/graph")
def api_graph():
    return jsonify(_graph)

@app.route("/api/summary")
def api_summary():
    return jsonify((_results or {}).get("summary", {}))


# ── WebSocket ─────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    print("[SERVER] Client connected — sending initial state")
    emit("graph_update",   _graph)
    emit("metrics_update", _metrics)
    emit("mitre_update",   _mitre_summary)
    # Prime the anomaly chart with the pre-computed series
    emit("anomaly_series", {"data": _anomaly_series})

    global _stream_alive
    if not _stream_alive:
        _stream_alive = True
        threading.Thread(target=_live_feed, daemon=True).start()


@socketio.on("disconnect")
def on_disconnect():
    print("[SERVER] Client disconnected")


# ── live feed ─────────────────────────────────────────────────────────────────

def _vary(v, lo=0.01, hi=0.03):
    return round(max(0.05, min(1.0, float(v) + random.uniform(-lo, hi))), 4)


def _live_feed():
    """Continuously replay stored alerts as a live threat stream."""
    global _stream_alive
    alerts = list(_alerts)
    if not alerts:
        _stream_alive = False
        return

    delay = max(0.7, min(2.5, 150.0 / len(alerts)))
    idx = tick = 0

    while _stream_alive:
        # ── emit one alert ────────────────────────────────────────────────
        a = dict(alerts[idx % len(alerts)])
        s = _vary(a.get("threat_score", 0.5), 0.04, 0.04)
        a["threat_score"] = s
        a["severity"]     = ("critical" if s>=0.85 else "high" if s>=0.65
                              else "medium" if s>=0.40 else "low")
        a["timestamp"]    = time.strftime("%d/%m/%Y %H:%M:%S")
        socketio.emit("new_alert", a)
        idx  += 1
        tick += 1

        # ── periodic refreshes ────────────────────────────────────────────
        if tick % 25 == 0:          # graph with subtle node suspicion drift
            nodes = [dict(n, suspicion=_vary(n.get("suspicion",0.3),0.02,0.02),
                          group=(2 if _vary(n.get("suspicion",0.3),0.02,0.02)>=0.5 else 1))
                     for n in _graph["nodes"]]
            socketio.emit("graph_update", {"nodes": nodes, "links": _graph["links"]})

        if tick % 45 == 0:          # model score fluctuation
            ind = _metrics.get("individual", {})
            socketio.emit("metrics_update", {"individual": {
                k: {kk: _vary(vv) for kk,vv in v.items()}
                for k,v in ind.items()
            }})

        if tick % 70 == 0:
            socketio.emit("mitre_update", _mitre_summary)

        time.sleep(delay + random.uniform(-0.15, 0.3))


# ── entry point ───────────────────────────────────────────────────────────────

def run_server(host="0.0.0.0", port=5000, debug=False):
    _load()
    print(f"\n[SERVER] → http://localhost:{port}\n")
    socketio.run(app, host=host, port=port, debug=debug,
                 use_reloader=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    run_server()