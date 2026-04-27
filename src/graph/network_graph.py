"""
graph/network_graph.py
----------------------
Graph-based threat intelligence.

Builds a directed graph where:
  - Nodes  = IP addresses
  - Edges  = detected attack flows (src_ip → dst_ip)

Each node receives a suspicion score computed from:
  - In-degree (how many attack flows target this IP)
  - Out-degree (how many attack flows originate from this IP)
  - PageRank centrality (how important is this IP in the attack network)
  - Number of unique attack types seen

Uses only NetworkX — no external graph ML library needed.
"""

import math
import numpy as np
import networkx as nx
from collections import defaultdict


# ─── Graph Builder ────────────────────────────────────────────────────────────

class AttackGraph:
    """
    Incremental directed attack graph with per-node suspicion scoring.
    """

    def __init__(self):
        self.G          = nx.DiGraph()
        self._edge_data = {}          # (src, dst) → list of flow dicts
        self._node_meta = defaultdict(lambda: {
            "attack_types": set(),
            "total_threat": 0.0,
            "flow_count":   0,
            "is_internal":  None,
        })

    # ── ingest ────────────────────────────────────────────────────────────────

    def add_flow(self, flow):
        """
        Add a single enriched flow dict to the graph.
        flow must have: source_ip, dest_ip, attack_type, threat_score.
        """
        src = flow.get("source_ip", "")
        dst = flow.get("dest_ip",   "")
        if not src or not dst or src == dst:
            return

        attack = flow.get("attack_type", "BENIGN")
        score  = float(flow.get("threat_score", 0))

        # Add / update nodes
        for ip, internal in [(src, _is_internal(src)), (dst, _is_internal(dst))]:
            if ip not in self.G:
                self.G.add_node(ip)
            meta = self._node_meta[ip]
            meta["is_internal"] = internal

        # Update node meta
        self._node_meta[src]["attack_types"].add(attack)
        self._node_meta[src]["total_threat"] += score
        self._node_meta[src]["flow_count"]   += 1

        self._node_meta[dst]["attack_types"].add(attack)
        self._node_meta[dst]["total_threat"] += score
        self._node_meta[dst]["flow_count"]   += 1

        # Add / update edge
        if self.G.has_edge(src, dst):
            self.G[src][dst]["weight"] += score
            self.G[src][dst]["count"]  += 1
            self.G[src][dst]["attacks"].add(attack)
        else:
            self.G.add_edge(src, dst, weight=score, count=1, attacks={attack})

        key = (src, dst)
        if key not in self._edge_data:
            self._edge_data[key] = []
        self._edge_data[key].append(flow)

    def add_batch(self, flows):
        """Add a list of enriched flows."""
        for f in flows:
            self.add_flow(f)

    # ── scoring ───────────────────────────────────────────────────────────────

    def compute_suspicion(self):
        """
        Compute a [0,1] suspicion score for every node.
        Stored as G.nodes[ip]['suspicion'].
        """
        if not self.G.nodes:
            return {}

        # PageRank (works on directed graphs naturally)
        try:
            pr = nx.pagerank(self.G, weight="weight", alpha=0.85, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            pr = {n: 1.0 / len(self.G) for n in self.G.nodes}

        # In-degree centraliry
        in_cent  = nx.in_degree_centrality(self.G)
        out_cent = nx.out_degree_centrality(self.G)

        scores = {}
        for node in self.G.nodes:
            meta      = self._node_meta[node]
            n_attacks = len(meta["attack_types"] - {"BENIGN"})
            avg_threat= (meta["total_threat"] / max(meta["flow_count"], 1))
            in_c      = in_cent.get(node, 0)
            out_c     = out_cent.get(node, 0)
            page_r    = pr.get(node, 0)

            # Weighted combination
            raw = (
                avg_threat * 0.35 +
                page_r     * 0.25 +
                out_c      * 0.20 +
                in_c       * 0.10 +
                min(1.0, n_attacks / 5.0) * 0.10
            )
            scores[node] = min(1.0, raw)
            self.G.nodes[node]["suspicion"]   = scores[node]
            self.G.nodes[node]["attack_types"]= list(meta["attack_types"])
            self.G.nodes[node]["is_internal"] = meta["is_internal"]
            self.G.nodes[node]["flow_count"]  = meta["flow_count"]

        return scores

    # ── export ────────────────────────────────────────────────────────────────

    def to_d3_format(self, max_nodes=60):
        """
        Export graph in D3.js force-graph format:
          { nodes: [{id, suspicion, group, ...}], links: [{source, target, ...}] }

        group=1 → normal host
        group=2 → attacker / suspicious
        """
        self.compute_suspicion()

        # Select top-N nodes by suspicion score
        node_scores = {n: self.G.nodes[n].get("suspicion", 0)
                       for n in self.G.nodes}
        top_nodes = sorted(node_scores, key=lambda x: -node_scores[x])[:max_nodes]
        top_set   = set(top_nodes)

        nodes_out = []
        for ip in top_nodes:
            data = self.G.nodes[ip]
            susp = data.get("suspicion", 0)
            nodes_out.append({
                "id":          ip,
                "suspicion":   round(susp, 4),
                "group":       2 if susp >= 0.5 else 1,
                "is_internal": data.get("is_internal", True),
                "flow_count":  data.get("flow_count", 0),
                "attack_types":data.get("attack_types", []),
            })

        links_out = []
        for src, dst, data in self.G.edges(data=True):
            if src in top_set and dst in top_set:
                attack_type = next(iter(data.get("attacks", {"unknown"})), "unknown")
                links_out.append({
                    "source":      src,
                    "target":      dst,
                    "value":       round(float(data.get("weight", 0)), 3),
                    "count":       int(data.get("count", 1)),
                    "attack_type": attack_type,
                })

        return {"nodes": nodes_out, "links": links_out}

    def top_attackers(self, n=10):
        """Return top-N nodes ordered by suspicion."""
        self.compute_suspicion()
        nodes = [(ip, self.G.nodes[ip].get("suspicion", 0))
                 for ip in self.G.nodes]
        return sorted(nodes, key=lambda x: -x[1])[:n]

    def summary_stats(self):
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "avg_degree":  (2 * self.G.number_of_edges() /
                            max(self.G.number_of_nodes(), 1)),
            "density":     nx.density(self.G) if self.G.number_of_nodes() > 1 else 0,
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_internal(ip):
    """Simple RFC-1918 check."""
    try:
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        a, b = int(parts[0]), int(parts[1])
        return (a == 10 or
                (a == 172 and 16 <= b <= 31) or
                (a == 192 and b == 168))
    except Exception:
        return False
