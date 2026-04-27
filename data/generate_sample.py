"""
generate_sample.py
------------------
Generates a realistic sample dataset that mirrors the CIC-IDS-2017 schema.
The real dataset can be downloaded from:
  https://www.unb.ca/cic/datasets/ids-2017.html

Drop the real CSV files into data/ and the pipeline will use them automatically.
"""

import csv
import random
import os

random.seed(42)

# CIC-IDS-2017 feature columns (subset of the 80 features)
COLUMNS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
    "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label"
]

ATTACK_TYPES = {
    "BENIGN":       0.45,
    "DoS Hulk":     0.08,
    "PortScan":     0.10,
    "DDoS":         0.08,
    "DoS GoldenEye":0.04,
    "FTP-Patator":  0.04,
    "SSH-Patator":  0.04,
    "DoS slowloris":0.03,
    "DoS Slowhttptest": 0.03,
    "Bot":          0.04,
    "Web Attack – Brute Force": 0.03,
    "Web Attack – XSS":         0.02,
    "Web Attack – Sql Injection":0.01,
    "Infiltration": 0.01,
}

INTERNAL_IPS = [f"192.168.{random.randint(0,3)}.{random.randint(1,254)}" for _ in range(20)]
EXTERNAL_IPS = [f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(30)]

def _rand_ip(internal=True):
    pool = INTERNAL_IPS if internal else EXTERNAL_IPS
    return random.choice(pool)

def _weighted_choice(d):
    keys = list(d.keys())
    weights = list(d.values())
    return random.choices(keys, weights=weights, k=1)[0]

def _flow_features(label):
    """Return dict of numerical features tuned per attack type."""
    base = {}

    if label == "BENIGN":
        base["Flow Duration"]          = random.randint(1000, 5000000)
        base["Total Fwd Packets"]      = random.randint(1, 50)
        base["Total Backward Packets"] = random.randint(0, 40)
        base["Flow Bytes/s"]           = random.uniform(100, 50000)
        base["Flow Packets/s"]         = random.uniform(0.5, 200)
        base["SYN Flag Count"]         = random.randint(0, 2)
        base["FIN Flag Count"]         = random.randint(0, 2)
        base["PSH Flag Count"]         = random.randint(0, 8)
        base["ACK Flag Count"]         = random.randint(0, 20)

    elif "DoS" in label or "DDoS" in label:
        base["Flow Duration"]          = random.randint(0, 500000)
        base["Total Fwd Packets"]      = random.randint(100, 5000)
        base["Total Backward Packets"] = random.randint(0, 10)
        base["Flow Bytes/s"]           = random.uniform(50000, 5000000)
        base["Flow Packets/s"]         = random.uniform(500, 50000)
        base["SYN Flag Count"]         = random.randint(0, 1000)
        base["FIN Flag Count"]         = 0
        base["PSH Flag Count"]         = random.randint(0, 5)
        base["ACK Flag Count"]         = random.randint(0, 5)

    elif "PortScan" in label:
        base["Flow Duration"]          = random.randint(0, 1000)
        base["Total Fwd Packets"]      = random.randint(1, 3)
        base["Total Backward Packets"] = random.randint(0, 1)
        base["Flow Bytes/s"]           = random.uniform(0, 1000)
        base["Flow Packets/s"]         = random.uniform(100, 10000)
        base["SYN Flag Count"]         = random.randint(1, 3)
        base["FIN Flag Count"]         = 0
        base["PSH Flag Count"]         = 0
        base["ACK Flag Count"]         = random.randint(0, 1)

    elif "Patator" in label:
        base["Flow Duration"]          = random.randint(500000, 3000000)
        base["Total Fwd Packets"]      = random.randint(10, 80)
        base["Total Backward Packets"] = random.randint(5, 60)
        base["Flow Bytes/s"]           = random.uniform(500, 8000)
        base["Flow Packets/s"]         = random.uniform(5, 80)
        base["SYN Flag Count"]         = random.randint(1, 5)
        base["FIN Flag Count"]         = random.randint(0, 3)
        base["PSH Flag Count"]         = random.randint(5, 30)
        base["ACK Flag Count"]         = random.randint(5, 40)

    else:
        base["Flow Duration"]          = random.randint(500, 2000000)
        base["Total Fwd Packets"]      = random.randint(2, 100)
        base["Total Backward Packets"] = random.randint(0, 80)
        base["Flow Bytes/s"]           = random.uniform(200, 80000)
        base["Flow Packets/s"]         = random.uniform(1, 400)
        base["SYN Flag Count"]         = random.randint(0, 3)
        base["FIN Flag Count"]         = random.randint(0, 2)
        base["PSH Flag Count"]         = random.randint(0, 20)
        base["ACK Flag Count"]         = random.randint(0, 30)

    # Fill remaining numeric columns with correlated random values
    for col in COLUMNS:
        if col in base or col in ("Flow ID", "Source IP", "Source Port",
                                   "Destination IP", "Destination Port",
                                   "Protocol", "Timestamp", "Label"):
            continue
        if "Variance" in col:
            base[col] = round(random.uniform(0, 50000), 4)
        elif "Std" in col:
            base[col] = round(random.uniform(0, 500), 4)
        elif "Mean" in col or "Avg" in col:
            base[col] = round(random.uniform(0, 1500), 4)
        elif "Max" in col:
            base[col] = random.randint(0, 65535)
        elif "Min" in col:
            base[col] = random.randint(0, 1000)
        elif "Flag" in col:
            base[col] = random.randint(0, 5)
        elif "bytes" in col.lower() or "Bytes" in col:
            base[col] = random.randint(0, 65535)
        else:
            base[col] = round(random.uniform(0, 1000), 4)

    return base


def generate(n_rows=5000, out_path=None):
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "sample_cicids2017.csv")

    from datetime import datetime, timedelta
    base_time = datetime(2017, 7, 7, 9, 0, 0)

    rows = []
    for i in range(n_rows):
        label = _weighted_choice(ATTACK_TYPES)
        is_external_src = label != "BENIGN"
        src_ip = _rand_ip(internal=not is_external_src)
        dst_ip = _rand_ip(internal=True)
        src_port = random.randint(1024, 65535)
        dst_port = random.choice([80, 443, 22, 21, 23, 3389, 8080, 53, 25])
        proto = random.choice([6, 17, 0])  # TCP/UDP/OTHER
        ts = base_time + timedelta(seconds=i * random.uniform(0.01, 2.0))
        flow_id = f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{proto}"

        feats = _flow_features(label)
        row = {
            "Flow ID": flow_id,
            "Source IP": src_ip,
            "Source Port": src_port,
            "Destination IP": dst_ip,
            "Destination Port": dst_port,
            "Protocol": proto,
            "Timestamp": ts.strftime("%d/%m/%Y %H:%M:%S"),
            **feats,
            "Label": label,
        }
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Generated {n_rows} rows → {out_path}")
    return out_path


if __name__ == "__main__":
    generate(5000)
