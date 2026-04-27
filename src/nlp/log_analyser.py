"""
nlp/log_analyser.py
-------------------
NLP-based log anomaly detection.

Approach (no external API, no pre-trained LLM):
  1. Convert network flow records into human-readable "log sentences"
  2. Build a TF-IDF vocabulary of known attack patterns
  3. Score new logs against known attack templates via cosine similarity
  4. Flag logs whose similarity exceeds a threshold
  5. Extract key entities (IPs, ports, attack tokens) using simple regex

This is deliberately self-contained: every computation is done
in pure Python + scikit-learn.
"""

import re
import math
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Attack signature templates ───────────────────────────────────────────────
# Each template is a short natural-language description of what a log entry
# from this attack type looks like. Used to build the TF-IDF reference corpus.

ATTACK_TEMPLATES = {
    "DoS Hulk": [
        "high volume http request flood short duration rapid packet transmission",
        "source sends massive fwd packets dest port 80 high bytes per second",
        "denial service hulk tool repeated get requests flood web server",
    ],
    "PortScan": [
        "sequential port sweep single source multiple destination ports",
        "fast low duration flows port scan reconnaissance probe",
        "syn flag sent no response port enumeration scanning activity",
    ],
    "DDoS": [
        "distributed denial service multiple sources flooding destination",
        "high packet rate coordinated attack saturate bandwidth",
        "many sources same destination high flow rate short duration ddos",
    ],
    "DoS GoldenEye": [
        "goldeneye slowread attack keep alive http persistent connection",
        "slow http attack tcp keep alive resource exhaustion web server",
    ],
    "FTP-Patator": [
        "brute force ftp login repeated authentication attempts different passwords",
        "ftp credential stuffing patator tool login failure repeated",
    ],
    "SSH-Patator": [
        "ssh brute force repeated login attempt port 22 credential bruteforce",
        "automated ssh authentication attack patator tool scanning credentials",
    ],
    "DoS slowloris": [
        "slowloris attack partial http headers connection held open resource exhaustion",
        "slow http dos low rate attack incomplete request headers timeout",
    ],
    "DoS Slowhttptest": [
        "slow http test post body slowly sent server resource exhaustion",
        "slowhttptest tool slow body attack web server dos",
    ],
    "Bot": [
        "bot activity periodic beaconing command control communication",
        "automated traffic regular intervals botnet c2 beacon callback",
    ],
    "Web Attack – Brute Force": [
        "web form brute force login repeated post request authentication",
        "http login page attack password guessing repeated credential test",
    ],
    "Web Attack – XSS": [
        "cross site scripting payload http get post script injection web",
        "xss reflected stored attack javascript inject web application",
    ],
    "Web Attack – Sql Injection": [
        "sql injection attack database query manipulation http parameter",
        "union select drop insert sql payload web application attack",
    ],
    "Infiltration": [
        "lateral movement internal network infiltration exploit vulnerability",
        "post exploitation internal traffic privilege escalation data exfil",
    ],
    "BENIGN": [
        "normal http web browsing low rate traffic standard ports",
        "benign regular user traffic tcp established acknowledged flow",
    ],
}

# MITRE ATT&CK mapping: attack_type → (technique_id, technique_name, tactic)
MITRE_MAP = {
    "PortScan":                       ("T1046", "Network Service Scanning",     "Discovery"),
    "DoS Hulk":                       ("T1499", "Endpoint Denial of Service",   "Impact"),
    "DDoS":                           ("T1498", "Network Denial of Service",    "Impact"),
    "DoS GoldenEye":                  ("T1499", "Endpoint Denial of Service",   "Impact"),
    "DoS slowloris":                  ("T1499.001", "OS Exhaustion Flood",      "Impact"),
    "DoS Slowhttptest":               ("T1499.001", "OS Exhaustion Flood",      "Impact"),
    "FTP-Patator":                    ("T1110", "Brute Force",                  "Credential Access"),
    "SSH-Patator":                    ("T1110.003", "Password Spraying",        "Credential Access"),
    "Bot":                            ("T1071", "Application Layer Protocol",   "Command and Control"),
    "Web Attack – Brute Force":       ("T1110", "Brute Force",                  "Credential Access"),
    "Web Attack – XSS":               ("T1059.007", "JavaScript",              "Execution"),
    "Web Attack – Sql Injection":     ("T1190", "Exploit Public-Facing App",    "Initial Access"),
    "Infiltration":                   ("T1021", "Remote Services",              "Lateral Movement"),
    "BENIGN":                         (None, None, None),
}

# Severity levels
def _severity(threat_score):
    if threat_score >= 0.85:
        return "critical"
    if threat_score >= 0.65:
        return "high"
    if threat_score >= 0.40:
        return "medium"
    return "low"


# ─── Simple tokeniser (no NLTK needed) ────────────────────────────────────────

def _tokenise(text):
    """Lowercase, split on non-alphanumeric, remove single-char tokens."""
    tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
    return tokens


def _flow_to_log_sentence(row):
    """
    Convert a flow record dict into a natural-language log string.
    This is the 'document' fed into TF-IDF.
    """
    parts = []

    src = str(row.get("source_ip", row.get("Source IP", "")))
    dst = str(row.get("dest_ip",   row.get("Destination IP", "")))
    sport = str(row.get("src_port", row.get("Source Port", "")))
    dport = str(row.get("dst_port", row.get("Destination Port", "")))
    proto = str(row.get("Protocol", ""))

    if src:
        parts.append(f"src {src}")
    if dst:
        parts.append(f"dst {dst}")
    if sport:
        parts.append(f"sport {sport}")
    if dport:
        dport_int = int(float(dport)) if dport.replace(".","").isdigit() else 0
        svc = _port_service(dport_int)
        parts.append(f"dport {dport} {svc}")
    if proto:
        p_int = int(float(proto)) if proto.replace(".","").isdigit() else 0
        parts.append(f"proto {_protocol_name(p_int)}")

    # Numeric features → descriptive tokens
    flow_dur = float(row.get("Flow Duration", 0) or 0)
    fwd_pkts = float(row.get("Total Fwd Packets", 0) or 0)
    bwd_pkts = float(row.get("Total Backward Packets", 0) or 0)
    flow_bps = float(row.get("Flow Bytes/s", 0) or 0)
    flow_pps = float(row.get("Flow Packets/s", 0) or 0)
    syn_flag = float(row.get("SYN Flag Count", 0) or 0)
    fin_flag = float(row.get("FIN Flag Count", 0) or 0)
    rst_flag = float(row.get("RST Flag Count", 0) or 0)
    psh_flag = float(row.get("PSH Flag Count", 0) or 0)

    if flow_dur < 1000:
        parts.append("short duration")
    elif flow_dur > 1000000:
        parts.append("long duration")

    if fwd_pkts > 500:
        parts.append("high volume fwd packets flood")
    elif fwd_pkts > 100:
        parts.append("elevated fwd packets")

    if flow_bps > 500000:
        parts.append("high bytes per second")
    if flow_pps > 1000:
        parts.append("high packet rate")

    if syn_flag > 50:
        parts.append("syn flood flag")
    if syn_flag > 0 and fin_flag == 0 and bwd_pkts < 2:
        parts.append("syn scan reconnaissance")
    if rst_flag > 10:
        parts.append("connection reset storm")
    if psh_flag > 30:
        parts.append("push flag heavy transmission")
    if fin_flag > 0:
        parts.append("connection teardown fin flag")

    return " ".join(parts)


def _port_service(port):
    mapping = {
        80: "http", 443: "https", 22: "ssh", 21: "ftp", 23: "telnet",
        25: "smtp", 53: "dns", 3389: "rdp", 8080: "http-alt",
        110: "pop3", 143: "imap", 3306: "mysql", 5432: "postgres",
        6379: "redis", 27017: "mongodb",
    }
    return mapping.get(port, "")


def _protocol_name(proto_num):
    return {6: "tcp", 17: "udp", 1: "icmp", 0: "other"}.get(proto_num, "other")


# ─── TF-IDF Analyser ──────────────────────────────────────────────────────────

class LogNLPAnalyser:
    """
    Builds TF-IDF model from attack templates, then scores new log entries
    via cosine similarity to detect and classify anomalies.
    """

    def __init__(self, similarity_threshold=0.25):
        self.threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            tokenizer=_tokenise,
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )
        self._labels   = []
        self._template_matrix = None
        self._fitted   = False

    def build_reference(self):
        """Fit TF-IDF on the attack template corpus."""
        corpus = []
        labels = []
        for attack_type, templates in ATTACK_TEMPLATES.items():
            for t in templates:
                corpus.append(t)
                labels.append(attack_type)

        self._labels = labels
        self.vectorizer.fit(corpus)
        self._template_matrix = self.vectorizer.transform(corpus)
        self._fitted = True
        print(f"  [LogNLPAnalyser] Reference built: {len(corpus)} templates, "
              f"{len(self.vectorizer.vocabulary_)} vocab tokens.")
        return self

    def analyse_flow(self, flow_dict):
        """
        Analyse a single flow record dict.
        Returns an enrichment dict with NLP fields.
        """
        if not self._fitted:
            raise RuntimeError("Call build_reference() first.")

        log_text = _flow_to_log_sentence(flow_dict)
        vec = self.vectorizer.transform([log_text])
        sims = cosine_similarity(vec, self._template_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_label = self._labels[best_idx]

        # Aggregate similarity per attack type
        type_scores = defaultdict(float)
        type_counts = defaultdict(int)
        for i, label in enumerate(self._labels):
            type_scores[label] = max(type_scores[label], float(sims[i]))
            type_counts[label] += 1

        top3 = sorted(type_scores.items(), key=lambda x: -x[1])[:3]

        # NLP-based threat score
        nlp_threat = min(1.0, best_sim * 1.5) if best_label != "BENIGN" else best_sim * 0.3

        # MITRE lookup
        mitre = MITRE_MAP.get(best_label, (None, None, None))

        # Human-readable explanation
        explanation = _explain(log_text, best_label, best_sim)

        return {
            "log_text":      log_text,
            "nlp_label":     best_label,
            "nlp_similarity":round(best_sim, 4),
            "nlp_threat":    round(nlp_threat, 4),
            "top3_matches":  [(l, round(s, 4)) for l, s in top3],
            "mitre": {
                "technique_id":   mitre[0],
                "technique_name": mitre[1],
                "tactic":         mitre[2],
            },
            "explanation":   explanation,
        }

    def analyse_batch(self, flow_dicts):
        """Analyse a list of flow dicts. Returns list of enrichment dicts."""
        return [self.analyse_flow(f) for f in flow_dicts]

    def enrich_detections(self, detections, flow_dicts):
        """
        Merge ML detector results with NLP analysis.
        detections: list of dicts from ThreatDetector.analyse()
        flow_dicts: corresponding raw row dicts
        Returns merged list.
        """
        enriched = []
        for det, flow in zip(detections, flow_dicts):
            nlp = self.analyse_flow(flow)

            # Blend threat scores (ML + NLP)
            ml_score  = det.get("threat_score", 0.0)
            nlp_score = nlp["nlp_threat"]
            combined  = round(ml_score * 0.65 + nlp_score * 0.35, 4)

            # Prefer ML attack label if confident, else use NLP label
            ml_label  = det.get("attack_type", "BENIGN")
            nlp_label = nlp["nlp_label"]
            if ml_label == "BENIGN" and nlp_label != "BENIGN":
                final_label = nlp_label
                combined = round(max(combined, nlp_score * 0.8), 4)
            else:
                final_label = ml_label

            mitre = MITRE_MAP.get(final_label, (None, None, None))

            merged = {
                **det,
                "attack_type":   final_label,
                "threat_score":  combined,
                "severity":      _severity(combined),
                "explanation":   nlp["explanation"],
                "log_text":      nlp["log_text"],
                "nlp_similarity":nlp["nlp_similarity"],
                "top3_matches":  nlp["top3_matches"],
                "mitre": {
                    "technique_id":   mitre[0],
                    "technique_name": mitre[1],
                    "tactic":         mitre[2],
                },
            }
            enriched.append(merged)

        return enriched


# ─── Explanation generator ────────────────────────────────────────────────────

def _explain(log_text, attack_type, similarity):
    """Generate a concise human-readable explanation."""
    base = {
        "DoS Hulk":     "High-rate HTTP flood detected. Abnormal packet volume and speed.",
        "PortScan":     "Sequential port sweep detected. Low duration flows across many ports.",
        "DDoS":         "Distributed flood from multiple sources. Bandwidth saturation pattern.",
        "DoS GoldenEye":"GoldenEye slow HTTP attack. Keep-alive abuse detected.",
        "FTP-Patator":  "FTP brute-force login attempts. Repeated auth failures.",
        "SSH-Patator":  "SSH credential stuffing. Rapid repeated auth attempts on port 22.",
        "DoS slowloris":"Slowloris attack. Connections held open to exhaust server resources.",
        "DoS Slowhttptest":"Slow HTTP body attack. Server resource exhaustion via slow POST.",
        "Bot":          "Botnet beacon detected. Periodic outbound C2 communication.",
        "Web Attack – Brute Force": "Web login brute force. Repeated POST to auth endpoint.",
        "Web Attack – XSS":         "XSS payload in HTTP parameters. Script injection attempt.",
        "Web Attack – Sql Injection":"SQL injection attempt. Malicious payload in query parameters.",
        "Infiltration": "Lateral movement detected. Internal network traversal behaviour.",
        "BENIGN":       "Traffic matches normal baseline. No anomaly detected.",
    }
    msg = base.get(attack_type, "Anomalous flow with elevated threat indicators.")
    conf = "High" if similarity > 0.6 else "Medium" if similarity > 0.3 else "Low"
    return f"{msg} (NLP confidence: {conf}, similarity: {similarity:.2f})"
