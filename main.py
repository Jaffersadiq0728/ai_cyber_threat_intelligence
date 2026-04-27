"""
main.py
-------
AI Cyber Threat Intelligence System
Entry point — runs the full pipeline end-to-end.

Usage:
    python main.py               # train + infer + report + start server
    python main.py --train-only  # train + infer + report (no server)
    python main.py --retrain     # force retrain even if cache exists
    python main.py --server-only # skip training, just serve results
    python main.py --rows 5000   # set how many flows to analyse (default 2000)

Real CIC-IDS-2017 data:
    1. Download from https://www.unb.ca/cic/datasets/ids-2017.html
    2. Place CSV files inside the data/ directory
    3. Re-run: python main.py --retrain
"""

import argparse
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)


def banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   AI CYBER THREAT INTELLIGENCE SYSTEM                           ║
║   Log NLP Anomaly Detection + Graph Learning                    ║
║   Dataset: CIC-IDS-2017                                         ║
╚══════════════════════════════════════════════════════════════════╝
""")


def ensure_sample_data():
    """Generate synthetic sample data if no real data exists."""
    data_dir = os.path.join(BASE_DIR, "data")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        print("[MAIN] No CSV data found. Generating synthetic sample data...")
        sys.path.insert(0, data_dir)
        from generate_sample import generate
        generate(5000, os.path.join(data_dir, "sample_cicids2017.csv"))
    else:
        print(f"[MAIN] Found {len(csv_files)} CSV file(s) in data/")


def run_pipeline(retrain=False, max_rows=2000):
    from engine.threat_engine import ThreatEngine

    engine = ThreatEngine(force_retrain=retrain)
    report = engine.run(max_rows=max_rows)

    # Print summary
    s = report.get("summary", {})
    m = report.get("model_metrics", {})
    print("\n" + "═" * 60)
    print("  RESULTS SUMMARY")
    print("═" * 60)
    print(f"  Total alerts  : {s.get('total_alerts', 0):,}")
    print(f"  Critical      : {s.get('critical', 0)}")
    print(f"  High          : {s.get('high', 0)}")
    print(f"  Medium        : {s.get('medium', 0)}")
    print(f"  Graph nodes   : {s.get('graph_nodes', 0)}")
    print(f"  Graph edges   : {s.get('graph_edges', 0)}")
    if m:
        print(f"  Accuracy      : {m.get('accuracy', 0):.4f}")
        print(f"  F1 (weighted) : {m.get('f1_weighted', 0):.4f}")
    print("═" * 60)

    # Print top attackers
    attackers = report.get("top_attackers", [])
    if attackers:
        print("\n  TOP SUSPICIOUS IPs:")
        for ip, score in attackers[:5]:
            bar = "█" * int(score * 20)
            print(f"  {ip:20s} {bar} {score:.3f}")

    print(f"\n  Report saved → reports/threat_report.json")
    print(f"  Results saved → output/results.json\n")
    return report


def main():
    banner()
    parser = argparse.ArgumentParser(
        description="AI Cyber Threat Intelligence System")
    parser.add_argument("--train-only",  action="store_true",
                        help="Train and analyse but do not start web server")
    parser.add_argument("--server-only", action="store_true",
                        help="Skip training, serve existing results")
    parser.add_argument("--retrain",     action="store_true",
                        help="Force model retraining")
    parser.add_argument("--rows",        type=int, default=2000,
                        help="Number of flows to analyse (default: 2000)")
    parser.add_argument("--port",        type=int, default=5000,
                        help="Dashboard server port (default: 5000)")
    args = parser.parse_args()

    if not args.server_only:
        ensure_sample_data()
        run_pipeline(retrain=args.retrain, max_rows=args.rows)

    if not args.train_only:
        print("[MAIN] Starting dashboard server...")
        print("[MAIN] Open your browser at  http://localhost:{}\n".format(args.port))
        sys.path.insert(0, BASE_DIR)
        from server import run_server
        run_server(port=args.port, debug=False)


if __name__ == "__main__":
    main()
