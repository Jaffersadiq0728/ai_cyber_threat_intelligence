# AI Cyber Threat Intelligence System
### Log NLP Anomaly Detection + Graph Learning | CIC-IDS-2017

A complete end-to-end cybersecurity AI pipeline that detects, classifies, and visualises
network intrusions using machine learning, NLP pattern matching, and graph analysis —
**no API keys, no external services, no pre-trained models**.

---

## Architecture

```
CIC-IDS-2017 CSV
      │
      ▼
┌─────────────────────────┐
│   Preprocessing          │  Strip, clean, scale, encode labels
│   cleaner.py             │
└──────────┬──────────────┘
           │
     ┌─────▼─────┐         ┌───────────────────────┐
     │  ML Models│         │  NLP Log Analyser      │
     │           │         │                        │
     │ Isolation │         │  Flow → log sentence   │
     │   Forest  │         │  TF-IDF vectorisation  │
     │ (anomaly) │         │  Cosine similarity     │
     │           │         │  to attack templates   │
     │  Random   │         │  MITRE ATT&CK mapping  │
     │  Forest   │         └───────────┬────────────┘
     │(classify) │                     │
     └─────┬─────┘                     │
           │        ◄──── MERGE ───────┘
           ▼
     ┌─────────────┐
     │ Attack Graph │  IP nodes, flow edges
     │  NetworkX    │  PageRank suspicion scoring
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Flask +     │  REST API + WebSocket
     │  Socket.IO   │  Live alert streaming
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Dashboard   │  D3.js force graph
     │  (HTML/JS)   │  Real-time alert feed
     └──────────────┘
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with synthetic data (no download needed)
```bash
python main.py
```

### 3. Open the dashboard
```
http://localhost:5000
```

---

## Using Real CIC-IDS-2017 Data

1. Download from: https://www.unb.ca/cic/datasets/ids-2017.html
2. Place the CSV files into the `data/` directory:
   ```
   data/
   ├── Monday-WorkingHours.pcap_ISCX.csv
   ├── Tuesday-WorkingHours.pcap_ISCX.csv
   ├── Wednesday-workingHours.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   ```
3. Retrain on the real data:
   ```bash
   python main.py --retrain --rows 50000
   ```

---

## Command Line Options

| Flag            | Description                                    |
|-----------------|------------------------------------------------|
| `--retrain`     | Force model retraining (ignore cache)          |
| `--train-only`  | Train + analyse, skip the web server           |
| `--server-only` | Skip training, serve existing results          |
| `--rows N`      | Number of flows to analyse (default: 2000)     |
| `--port N`      | Dashboard server port (default: 5000)          |

---

## Project Structure

```
ai_cyber_threat_intelligence/
├── main.py                        ← Entry point
├── server.py                      ← Flask + Socket.IO server
├── requirements.txt
│
├── data/
│   ├── generate_sample.py         ← Synthetic data generator
│   └── sample_cicids2017.csv      ← Generated sample (auto-created)
│
├── src/
│   ├── preprocessing/
│   │   └── cleaner.py             ← Data loading, cleaning, feature engineering
│   │
│   ├── models/
│   │   └── anomaly_detector.py    ← Isolation Forest + Random Forest
│   │
│   ├── nlp/
│   │   └── log_analyser.py        ← TF-IDF NLP + MITRE ATT&CK mapping
│   │
│   ├── graph/
│   │   └── network_graph.py       ← NetworkX attack graph + suspicion scoring
│   │
│   └── engine/
│       └── threat_engine.py       ← Orchestration pipeline
│
├── dashboard/
│   └── index.html                 ← Live threat intelligence dashboard
│
├── output/
│   ├── results.json               ← Full inference results
│   └── threat_detector.pkl        ← Cached model
│
└── reports/
    └── threat_report.json         ← Human-readable threat report
```

---

## Models

### 1. Isolation Forest (Anomaly Scoring)
- Trained **only on benign traffic** to learn the normal baseline
- Assigns each flow an anomaly score in [0, 1]
- Does NOT require labelled data — works as unsupervised detector

### 2. Random Forest Classifier (Attack Classification)
- Trained on full labelled CIC-IDS-2017 dataset
- 14-class classification (BENIGN + 13 attack types)
- `class_weight='balanced'` handles class imbalance
- Feature importances available for explainability

### 3. NLP Log Analyser (Pattern Matching)
- Converts raw flow records into human-readable "log sentences"
- Builds TF-IDF vocabulary from curated attack template corpus
- Cosine similarity scoring against known attack patterns
- Provides MITRE ATT&CK technique + tactic mapping
- Generates human-readable explanations for each alert

### 4. Attack Graph (Threat Intelligence)
- Directed graph: IP nodes, flow edges weighted by threat score
- PageRank centrality to identify key attackers
- In/out-degree analysis for victim vs. attacker classification
- Suspicion scoring: combined PageRank + avg threat + unique attack types
- Exported as D3.js force graph for the dashboard

---

## Dashboard Features

- **Live Alert Feed** — real-time stream with severity colour-coding
- **Anomaly Score Chart** — threshold-annotated time series
- **D3 Force Graph** — interactive attack network with draggable nodes
- **MITRE ATT&CK Heatmap** — tactics frequency breakdown
- **Model Performance Meters** — accuracy and F1 score display
- **Alert Detail Drawer** — click any alert to see full enrichment

---

## Attack Types Detected

| Attack           | MITRE Technique     | Tactic              |
|------------------|---------------------|---------------------|
| PortScan         | T1046               | Discovery           |
| DoS Hulk         | T1499               | Impact              |
| DDoS             | T1498               | Impact              |
| FTP-Patator      | T1110               | Credential Access   |
| SSH-Patator      | T1110.003           | Credential Access   |
| Bot              | T1071               | Command & Control   |
| Web Attack – XSS | T1059.007           | Execution           |
| SQL Injection    | T1190               | Initial Access      |
| Infiltration     | T1021               | Lateral Movement    |
| DoS slowloris    | T1499.001           | Impact              |

---

## No External Dependencies Beyond Listed Packages

- ✅ No OpenAI / Claude / any LLM API calls
- ✅ No pre-trained model downloads
- ✅ No internet connection required at inference time
- ✅ All algorithms implemented from scratch using scikit-learn primitives
- ✅ 100% locally executable
