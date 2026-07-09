# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
KnowledgeGraph
================
Weighted, degree-capped, dedup'd in-memory graph with BFS / DFS / A*
traversal and triple-dedup based on (subject, relation, object). Ported
1:1 from engine_v1.py's ``KnowledgeGraph`` (including the KG-FIX commits:
degree-cap always removes the reverse edge; triple dedup key includes
the relation for richer semantics).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..constants import KG_MAX_DEGREE, KG_MIN_EDGE_WEIGHT
from ..utils.hashing import stable_hash
from .graph_search import astar as _astar
from .graph_search import bfs as _bfs
from .graph_search import dfs as _dfs


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g: Dict[str, Dict[str, float]] = {}
        self._triples: List[Tuple[str, str, str, float]] = []
        self._triple_hashes: set = set()

    # ── Internal helpers ──────────────────────────────────────────────
    @staticmethod
    def _triple_key(subj: str, rel: str, obj: str) -> str:
        """Includes the relation in the dedup key so the same (subj, obj)
        pair under different relations is treated as a distinct triple —
        preserves richer graph semantics."""
        return stable_hash(subj, rel, obj)

    def _enforce_degree_cap(self, node: str) -> None:
        """Remove lowest-weight edges if node exceeds max degree. Always
        removes the reverse edge when a forward edge is pruned, keeping
        the graph symmetric (no orphaned back-edges)."""
        neighbors = self.g.get(node, {})
        if len(neighbors) > KG_MAX_DEGREE:
            sorted_nbs = sorted(neighbors.items(), key=lambda x: x[1])
            to_drop = len(neighbors) - KG_MAX_DEGREE
            for nb, _ in sorted_nbs[:to_drop]:
                del self.g[node][nb]
                self.g.get(nb, {}).pop(node, None)

    def _prune_weak_edges(self, node: str) -> None:
        neighbors = self.g.get(node, {})
        weak = [nb for nb, w in neighbors.items() if w < KG_MIN_EDGE_WEIGHT]
        for nb in weak:
            del self.g[node][nb]

    # ── Public API ────────────────────────────────────────────────────
    def add(self, a: str, b: str, w: float = 1.0) -> None:
        self.g.setdefault(a, {})[b] = w
        self.g.setdefault(b, {})[a] = w
        self._prune_weak_edges(a)
        self._prune_weak_edges(b)
        self._enforce_degree_cap(a)
        self._enforce_degree_cap(b)

    def bfs(self, src: str, dst: str) -> Optional[List[str]]:
        return _bfs(self.g, src, dst)

    def dfs(self, src: str, dst: str, max_d: int = 6) -> Optional[List[str]]:
        return _dfs(self.g, src, dst, max_d)

    def astar(self, src: str, dst: str) -> Tuple[Optional[List[str]], float]:
        return _astar(self.g, src, dst)

    def related(self, concept: str, hops: int = 2) -> Dict[str, int]:
        from collections import deque

        res: Dict[str, int] = {}
        q = deque([(concept, 0)])
        while q:
            n, d = q.popleft()
            if n in res or d > hops:
                continue
            res[n] = d
            for nb in self.g.get(n, {}):
                q.append((nb, d + 1))
        res.pop(concept, None)
        return res

    # ── Dynamic triple extraction from conversation ──────────────────
    def extract_and_add_triples(self, text: str, conf: float = 0.6) -> int:
        """Delegates pattern extraction to graph_extractor.TripleExtractor
        and adds the resulting triples into this graph. Returns count added."""
        from .graph_extractor import TripleExtractor

        extractor = TripleExtractor()
        added = 0
        for subj, rel, obj, w in extractor.extract(text, conf):
            key = self._triple_key(subj, rel, obj)
            if key in self._triple_hashes:
                continue
            self._triple_hashes.add(key)
            self.add(subj, obj, w)
            self._triples.append((subj, rel, obj, w))
            added += 1
            if added >= 5:  # cap per call
                break
        return added

    def init_default(self) -> None:
        """DracoAI branding edges used throughout the demo graph."""
        edges = [
            ("AI", "Machine Learning", 0.9), ("Machine Learning", "Deep Learning", 0.8),
            ("Deep Learning", "Transformer", 0.9), ("Transformer", "Attention", 0.95),
            ("Transformer", "DracoAI", 0.9), ("Transformer", "DeepSeek", 0.8),
            ("Attention", "GQA", 0.8), ("GQA", "KV Cache", 0.8),
            ("MoE", "Transformer", 0.8), ("Mirostat", "Sampling", 0.9),
            ("Python", "NumPy", 0.8), ("Python", "AI", 0.7),
            ("Embedding", "Vector", 0.95), ("Token", "Embedding", 0.9),
            ("RoPE", "Positional Encoding", 0.9),
            ("BFS", "Graph", 0.9), ("DFS", "Graph", 0.9),
            ("A*", "Graph", 0.9), ("MCTS", "Tree Search", 0.9),
            ("ToT", "Reasoning", 0.9), ("DracoAI", "MoE", 0.95),
            ("DracoAI", "SwiGLU", 0.8), ("Code", "Python", 0.8),
            ("Code", "Debug", 0.7), ("DracoAI", "Qwen 3.5 9B", 0.85),
            ("DracoAI", "Identity Overlay", 0.9),
            ("Qwen 3.5 9B", "MoE", 0.7), ("Qwen 3.5 9B", "SwiGLU", 0.7),
        ]
        for a, b, w in edges:
            self.add(a, b, w)
