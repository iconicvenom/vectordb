import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from scipy.spatial import KDTree as ScipyKDTree

DIM = 16
DOC_DIM = 768
SERVER_START = time.time()

BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "hnsw_index.bin"
DOC_INDEX_PATH = BASE_DIR / "doc_hnsw_index.bin"

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2"


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a_norm, b_norm))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


DISTANCE_FNS = {
    "cosine": cosine_distance,
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


class BruteForce:
    def __init__(self):
        self.items: dict[int, dict[str, Any]] = {}

    def insert(self, item_id: int, vector: np.ndarray, label: str, category: str, metadata: dict | None = None):
        self.items[item_id] = {
            "id": item_id,
            "vector": vector.astype(np.float32),
            "label": label,
            "category": category,
            "metadata": metadata or {},
        }

    def delete(self, item_id: int) -> bool:
        return self.items.pop(item_id, None) is not None

    def list_all(self) -> list[dict]:
        return [
            {"id": v["id"], "label": v["label"], "category": v["category"], "metadata": v["metadata"]}
            for v in self.items.values()
        ]

    def search(self, query: np.ndarray, k: int, metric: str = "cosine") -> list[dict]:
        fn = DISTANCE_FNS.get(metric, cosine_distance)
        scored = []
        for item in self.items.values():
            dist = fn(query, item["vector"])
            scored.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "distance": round(dist, 4),
                    "metadata": item["metadata"],
                    "category": item["category"],
                }
            )
        scored.sort(key=lambda x: x["distance"])
        return scored[:k]


class KDTree:
    """Wraps scipy.spatial.KDTree; Euclidean distance only."""

    def __init__(self):
        self.items: dict[int, dict[str, Any]] = {}
        self._tree: ScipyKDTree | None = None
        self._id_order: list[int] = []

    def _rebuild(self):
        if not self.items:
            self._tree = None
            self._id_order = []
            return
        self._id_order = list(self.items.keys())
        vectors = np.array([self.items[i]["vector"] for i in self._id_order], dtype=np.float32)
        self._tree = ScipyKDTree(vectors)

    def insert(self, item_id: int, vector: np.ndarray, label: str, category: str, metadata: dict | None = None):
        self.items[item_id] = {
            "id": item_id,
            "vector": vector.astype(np.float32),
            "label": label,
            "category": category,
            "metadata": metadata or {},
        }
        self._rebuild()

    def delete(self, item_id: int) -> bool:
        if item_id not in self.items:
            return False
        del self.items[item_id]
        self._rebuild()
        return True

    def list_all(self) -> list[dict]:
        return [
            {"id": v["id"], "label": v["label"], "category": v["category"], "metadata": v["metadata"]}
            for v in self.items.values()
        ]

    def search(self, query: np.ndarray, k: int, metric: str = "euclidean") -> list[dict]:
        if not self.items or self._tree is None:
            return []
        dists, indices = self._tree.query(query.astype(np.float32), k=min(k, len(self.items)))
        if np.isscalar(indices):
            indices = [int(indices)]
            dists = [float(dists)]
        else:
            indices = [int(i) for i in indices]
            dists = [float(d) for d in dists]
        results = []
        for idx, dist in zip(indices, dists):
            item_id = self._id_order[idx]
            item = self.items[item_id]
            results.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "distance": round(dist, 4),
                    "metadata": item["metadata"],
                    "category": item["category"],
                }
            )
        return results


class HNSWIndex:
    def __init__(self, dim: int, max_elements: int = 10000, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.dim = dim
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(ef_search)
        self.items: dict[int, dict[str, Any]] = {}
        self._next_slot = 0
        self._slot_to_id: dict[int, int] = {}
        self._id_to_slot: dict[int, int] = {}

    def insert(self, item_id: int, vector: np.ndarray, label: str, category: str, metadata: dict | None = None):
        if item_id in self._id_to_slot:
            slot = self._id_to_slot[item_id]
        else:
            slot = self._next_slot
            self._next_slot += 1
            self._slot_to_id[slot] = item_id
            self._id_to_slot[item_id] = slot
        vec = _normalize(vector)
        if slot >= self.index.get_current_count():
            self.index.add_items(vec.reshape(1, -1), np.array([slot]))
        else:
            self.index.mark_deleted(slot)
            self.index.add_items(vec.reshape(1, -1), np.array([slot]))
        self.items[item_id] = {
            "id": item_id,
            "vector": vec,
            "label": label,
            "category": category,
            "metadata": metadata or {},
        }

    def delete(self, item_id: int) -> bool:
        if item_id not in self._id_to_slot:
            return False
        slot = self._id_to_slot[item_id]
        self.index.mark_deleted(slot)
        del self._id_to_slot[item_id]
        del self._slot_to_id[slot]
        del self.items[item_id]
        return True

    def search(self, query: np.ndarray, k: int, metric: str = "cosine") -> list[dict]:
        if not self.items:
            return []
        q = _normalize(query)
        labels, distances = self.index.knn_query(q.reshape(1, -1), k=min(k, len(self.items)))
        results = []
        for slot, dist in zip(labels[0], distances[0]):
            slot = int(slot)
            if slot not in self._slot_to_id:
                continue
            item_id = self._slot_to_id[slot]
            if item_id not in self.items:
                continue
            item = self.items[item_id]
            results.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "distance": round(float(dist), 4),
                    "metadata": item["metadata"],
                    "category": item["category"],
                }
            )
        return results

    def info(self) -> dict:
        return {
            "total_elements": len(self.items),
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }


def _make_demo_vectors() -> list[dict]:
    """Handcrafted 16D vectors clustered by category."""
    rng = np.random.default_rng(42)
    clusters = {
        "CS": np.array([1.0, 0.9, 0.85, 0.1, 0.05, 0.0, 0.2, 0.15, 0.0, 0.1, 0.05, 0.0, 0.0, 0.1, 0.05, 0.0]),
        "Math": np.array([0.1, 0.15, 0.0, 1.0, 0.95, 0.9, 0.05, 0.0, 0.1, 0.0, 0.05, 0.0, 0.0, 0.0, 0.1, 0.05]),
        "Food": np.array([0.0, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.05, 1.0, 0.95, 0.9, 0.85, 0.1, 0.0, 0.05, 0.0]),
        "Sports": np.array([0.05, 0.0, 0.1, 0.05, 0.0, 0.1, 0.05, 0.0, 0.1, 0.05, 0.0, 0.1, 0.95, 0.9, 0.85, 0.8]),
    }
    labels = {
        "CS": [
            "binary tree", "linked list", "hash table", "graph traversal",
            "dynamic programming", "sorting algorithm", "B-tree", "red-black tree",
            "Dijkstra shortest path", "A* search", "merge sort", "quick sort",
        ],
        "Math": [
            "linear algebra", "calculus", "probability", "number theory", "topology",
            "group theory", "real analysis", "combinatorics", "differential equations", "set theory",
        ],
        "Food": [
            "pasta carbonara", "sushi roll", "tacos al pastor", "pad thai", "croissant",
            "ramen bowl", "margherita pizza", "bibimbap", "chicken biryani", "falafel wrap",
        ],
        "Sports": [
            "basketball", "soccer", "tennis", "swimming",
            "volleyball", "baseball", "marathon", "rock climbing",
        ],
    }
    counts = {cat: len(labels[cat]) for cat in labels}
    items = []
    item_id = 1
    for cat, count in counts.items():
        base = clusters[cat]
        for i in range(count):
            noise = rng.normal(0, 0.025, DIM)
            vec = _normalize(base + noise)
            items.append(
                {
                    "id": item_id,
                    "label": labels[cat][i],
                    "vector": vec,
                    "category": cat,
                    "metadata": {"source": "demo"},
                }
            )
            item_id += 1
    return items


class VectorDB:
    def __init__(self):
        self.brute = BruteForce()
        self.kdtree = KDTree()
        self.hnsw = HNSWIndex(dim=DIM)
        self.vectors: dict[int, np.ndarray] = {}
        for item in _make_demo_vectors():
            self.insert(item["id"], item["label"], item["vector"], item["category"], item.get("metadata"))

    def insert(self, item_id: int, label: str, vector: np.ndarray, category: str, metadata: dict | None = None):
        vec = vector.astype(np.float32)
        self.vectors[item_id] = vec
        self.brute.insert(item_id, vec, label, category, metadata)
        self.kdtree.insert(item_id, vec, label, category, metadata)
        self.hnsw.insert(item_id, vec, label, category, metadata)

    def delete(self, item_id: int) -> bool:
        self.vectors.pop(item_id, None)
        b = self.brute.delete(item_id)
        k = self.kdtree.delete(item_id)
        h = self.hnsw.delete(item_id)
        return b or k or h

    def search(self, query: np.ndarray, k: int, metric: str, algo: str) -> list[dict]:
        q = query.astype(np.float32)
        if algo == "hnsw":
            return self.hnsw.search(q, k, metric)
        if algo == "kdtree":
            return self.kdtree.search(q, k, metric)
        return self.brute.search(q, k, metric)

    def list_all(self) -> list[dict]:
        return self.brute.list_all()

    def all_vectors_matrix(self) -> tuple[np.ndarray, list[dict]]:
        items = self.brute.list_all()
        ids = sorted(self.vectors.keys())
        matrix = np.array([self.vectors[i] for i in ids], dtype=np.float32)
        meta = []
        id_to_item = {it["id"]: it for it in items}
        for i in ids:
            it = id_to_item[i]
            meta.append({"id": i, "label": it["label"], "category": it["category"]})
        return matrix, meta

    def benchmark(self, query: np.ndarray, k: int, metric: str) -> dict[str, float]:
        q = query.astype(np.float32)
        results = {}
        for name, algo in [("hnsw", "hnsw"), ("kdtree", "kdtree"), ("brute", "brute")]:
            start = time.perf_counter()
            for _ in range(50):
                self.search(q, k, metric, algo)
            elapsed = (time.perf_counter() - start) / 50 * 1000
            results[name] = round(elapsed, 2)
        return results

    def benchmark_sweep(self, query: np.ndarray, metric: str) -> dict:
        sweep = {}
        for k_val in [1, 5, 10, 20, 50]:
            sweep[str(k_val)] = self.benchmark(query, k_val, metric)
        return sweep

    def pca_coords(self) -> dict:
        matrix, meta = self.all_vectors_matrix()
        if len(matrix) == 0:
            return {"points": [], "variance": {"pc1": 0, "pc2": 0, "pc3": 0}}
        centered = matrix - matrix.mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt.T
        total_var = np.sum(s ** 2)
        var_pct = [(s[i] ** 2 / total_var * 100) if total_var > 0 else 0 for i in range(min(3, len(s)))]
        while len(var_pct) < 3:
            var_pct.append(0.0)
        scale = 1.2
        points = []
        for i, m in enumerate(meta):
            points.append(
                {
                    "id": m["id"],
                    "label": m["label"],
                    "category": m["category"],
                    "x": round(float(projected[i, 0] * scale), 4),
                    "y": round(float(projected[i, 1] * scale) if projected.shape[1] > 1 else 0, 4),
                    "z": round(float(projected[i, 2] * scale) if projected.shape[1] > 2 else 0, 4),
                }
            )
        points = _layout_quadrants(points)
        edges = _build_cluster_edges(points)
        return {
            "points": points,
            "edges": edges,
            "variance": {
                "pc1": round(float(var_pct[0]), 1),
                "pc2": round(float(var_pct[1]), 1),
                "pc3": round(float(var_pct[2]), 1),
            },
        }


def _layout_quadrants(points: list[dict]) -> list[dict]:
    """Place each category in a quadrant as a compact circular cluster."""
    centers = {
        "CS": (-5.5, 5.5, 2.4),
        "Sports": (5.5, 5.5, 2.0),
        "Math": (-5.5, -5.5, -2.0),
        "Food": (5.5, -5.5, -2.4),
    }
    by_cat: dict[str, list[dict]] = {}
    for p in points:
        by_cat.setdefault(p["category"], []).append(p)

    laid_out: list[dict] = []
    for cat, cat_points in by_cat.items():
        cx, cy, cz = centers.get(cat, (0.0, 0.0, 0.0))
        n = len(cat_points)
        base_r = 0.9 + min(n, 24) * 0.045
        for i, p in enumerate(cat_points):
            angle = (2 * np.pi * i / n) - (np.pi / 2)
            ring = 0.75 + (i % 4) * 0.12
            r = base_r * ring
            wobble = 0.08 * np.sin(i * 1.9 + hash(cat) % 7)
            laid_out.append(
                {
                    **p,
                    "x": round(float(cx + (r + wobble) * np.cos(angle)), 4),
                    "y": round(float(cy + (r + wobble) * np.sin(angle)), 4),
                    "z": round(float(cz + p.get("z", 0) * 0.4), 4),
                }
            )
    return laid_out


def _build_cluster_edges(points: list[dict]) -> dict:
    """Build dense intra-cluster mesh + hub spokes for network visualization."""
    by_cat: dict[str, list[dict]] = {}
    for p in points:
        by_cat.setdefault(p["category"], []).append(p)

    edges: list[dict] = []
    for cat_points in by_cat.values():
        n = len(cat_points)
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                dx = cat_points[i]["x"] - cat_points[j]["x"]
                dy = cat_points[i]["y"] - cat_points[j]["y"]
                dz = cat_points[i]["z"] - cat_points[j]["z"]
                dists.append((j, dx * dx + dy * dy + dz * dz))
            dists.sort(key=lambda x: x[1])
            connect_to = min(n - 1, 3) if n > 4 else n - 1
            seen: set[int] = set()
            for j, _ in dists[:connect_to]:
                if j in seen:
                    continue
                seen.add(j)
                a, b = cat_points[i]["id"], cat_points[j]["id"]
                if a < b:
                    edges.append({"from": a, "to": b, "category": cat_points[i]["category"]})

    hub = {"x": 0.0, "y": 0.0, "z": 0.0}
    hub_edges: list[dict] = []
    for cat, cat_points in by_cat.items():
        anchor = min(cat_points, key=lambda p: p["x"] ** 2 + p["y"] ** 2 + p["z"] ** 2)
        hub_edges.append({"from": "hub", "to": anchor["id"], "category": cat})
        if len(cat_points) > 1:
            secondary = sorted(
                cat_points,
                key=lambda p: (p["x"] - anchor["x"]) ** 2 + (p["y"] - anchor["y"]) ** 2,
            )[1]
            hub_edges.append({"from": "hub", "to": secondary["id"], "category": cat})

    return {"intra": edges, "hub": hub, "hub_edges": hub_edges}


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE):
        self.base_url = base_url.rstrip("/")
        self.embed_model = EMBED_MODEL
        self.gen_model = GEN_MODEL

    def is_online(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return []
            data = r.json()
            return [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        except Exception:
            return []

    def embed(self, text: str) -> np.ndarray:
        r = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        emb = r.json().get("embedding", [])
        if len(emb) != DOC_DIM:
            vec = np.zeros(DOC_DIM, dtype=np.float32)
            copy_len = min(len(emb), DOC_DIM)
            vec[:copy_len] = emb[:copy_len]
            return _normalize(vec)
        return _normalize(np.array(emb, dtype=np.float32))

    def generate(self, prompt: str) -> str:
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.gen_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()


class DocumentDB:
    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama
        self.hnsw = HNSWIndex(dim=DOC_DIM, max_elements=50000)
        self.documents: dict[str, dict] = {}
        self.chunks: dict[int, dict] = {}
        self._next_chunk_id = 10000

    def chunk_text(self, text: str, size: int = 250, overlap: int = 50) -> list[str]:
        text = text.strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    def insert(self, title: str, text: str) -> dict:
        doc_id = str(uuid.uuid4())[:8]
        chunks = self.chunk_text(text)
        chunk_records = []
        for idx, chunk in enumerate(chunks):
            try:
                vec = self.ollama.embed(chunk)
            except Exception:
                vec = np.zeros(DOC_DIM, dtype=np.float32)
                vec[idx % DOC_DIM] = 1.0
                vec = _normalize(vec)
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            meta = {"doc_id": doc_id, "title": title, "chunk_index": idx, "preview": chunk[:120]}
            self.hnsw.insert(chunk_id, vec, f"{title} [{idx}]", "Custom", meta)
            self.chunks[chunk_id] = {"id": chunk_id, "doc_id": doc_id, "text": chunk, **meta}
            chunk_records.append({"chunk_id": chunk_id, "chunk_index": idx, "preview": chunk[:120]})
        self.documents[doc_id] = {"id": doc_id, "title": title, "chunks": len(chunks), "created": datetime.utcnow().isoformat()}
        return {"doc_id": doc_id, "title": title, "chunks": chunk_records}

    def list_docs(self) -> list[dict]:
        return list(self.documents.values())

    def delete_doc(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        to_delete = [cid for cid, c in self.chunks.items() if c["doc_id"] == doc_id]
        for cid in to_delete:
            self.hnsw.delete(cid)
            del self.chunks[cid]
        del self.documents[doc_id]
        return True

    def ask(self, question: str, k: int = 3) -> dict:
        if not self.chunks:
            return {"answer": "No documents have been embedded yet. Please insert documents first.", "sources": []}
        try:
            q_vec = self.ollama.embed(question)
        except Exception:
            return {"answer": "Could not embed question — is Ollama running?", "sources": []}
        results = self.hnsw.search(q_vec, k)
        context_parts = []
        sources = []
        for r in results:
            meta = r.get("metadata", {})
            preview = meta.get("preview", r["label"])
            context_parts.append(f"[{meta.get('title', 'Doc')}] {preview}")
            sources.append(
                {
                    "title": meta.get("title", "Unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "preview": preview,
                    "distance": r["distance"],
                }
            )
        context = "\n\n".join(context_parts)
        prompt = (
            f"You are a helpful assistant. Answer the question based on the following document excerpts.\n\n"
            f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer concisely with bullet points:"
        )
        try:
            answer = self.ollama.generate(prompt)
        except Exception as e:
            answer = f"Could not generate response: {e}"
        return {"answer": answer, "sources": sources}


def get_stats(vdb: VectorDB, docdb: DocumentDB) -> dict:
    demo_count = len(vdb.vectors)
    doc_count = len(docdb.chunks)
    total = demo_count + doc_count
    dims = DOC_DIM if doc_count > 0 else DIM
    memory_bytes = demo_count * DIM * 4 + doc_count * DOC_DIM * 4
    hnsw_size = (demo_count + doc_count) * dims * 4 * 0.5
    return {
        "total_vectors": total,
        "dimensions": dims,
        "collections": 1 + (1 if doc_count > 0 else 0),
        "memory_mb": round(memory_bytes / (1024 * 1024), 1),
        "hnsw_size_mb": round(hnsw_size / (1024 * 1024), 1),
        "uptime_seconds": int(time.time() - SERVER_START),
    }


class InsertRequest(BaseModel):
    id: int
    label: str
    vector: list[float]
    category: str = "Custom"


class DocInsertRequest(BaseModel):
    title: str
    text: str


class DocAskRequest(BaseModel):
    question: str
    k: int = 3


app = FastAPI(title="VectorDB", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama = OllamaClient()
vdb = VectorDB()
docdb = DocumentDB(ollama)
_last_updated = time.time()


@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")


def _demo_query_vector(text: str) -> np.ndarray:
    """Map search text to demo 16D space via label/category matching."""
    text_lower = text.lower()
    best_id = None
    best_score = -1
    for item in vdb.brute.list_all():
        label = item["label"].lower()
        score = 0
        for word in text_lower.split():
            if word in label or label in text_lower:
                score += 2
            if word in label.split():
                score += 1
        if item["category"] == "CS" and any(w in text_lower for w in ("tree", "algorithm", "code", "graph", "hash", "sort")):
            score += 1
        if score > best_score:
            best_score = score
            best_id = item["id"]
    if best_id is not None and best_score > 0:
        base = vdb.vectors[best_id].copy()
    else:
        rng = np.random.default_rng(hash(text) % 2**32)
        base = rng.normal(0, 1, DIM).astype(np.float32)
    noise = np.random.default_rng(hash(text + "n") % 2**32).normal(0, 0.02, DIM)
    return _normalize(base + noise)


@app.get("/search")
def search(
    v: str = Query(...),
    k: int = 10,
    metric: str = "cosine",
    algo: str = "hnsw",
):
    global _last_updated
    q_vec = _demo_query_vector(v)
    results = vdb.search(q_vec, k, metric, algo)
    _last_updated = time.time()
    return {"query": v, "results": results, "algo": algo, "metric": metric}


@app.post("/insert")
def insert_item(req: InsertRequest):
    global _last_updated
    vec = np.array(req.vector, dtype=np.float32)
    if len(vec) != DIM:
        raise HTTPException(400, f"Vector must be {DIM} dimensions")
    vdb.insert(req.id, req.label, vec, req.category)
    _last_updated = time.time()
    return {"ok": True, "id": req.id}


@app.delete("/delete/{item_id}")
def delete_item(item_id: int):
    global _last_updated
    ok = vdb.delete(item_id)
    _last_updated = time.time()
    if not ok:
        raise HTTPException(404, "Item not found")
    return {"ok": True}


@app.get("/items")
def list_items():
    return {"items": vdb.list_all()}


@app.get("/benchmark")
def benchmark(
    v: str = Query(...),
    k: int = 10,
    metric: str = "cosine",
    sweep: bool = False,
):
    q_vec = _demo_query_vector(v)
    if sweep:
        return vdb.benchmark_sweep(q_vec, metric)
    return vdb.benchmark(q_vec, k, metric)


@app.get("/hnsw-info")
def hnsw_info():
    return vdb.hnsw.info()


@app.get("/stats")
def stats():
    return get_stats(vdb, docdb)


@app.get("/pca-coords")
def pca_coords():
    return vdb.pca_coords()


@app.get("/models")
def models():
    return {"models": ollama.list_models()}


@app.get("/status")
def status():
    return {
        "ollama_online": ollama.is_online(),
        "embed_model": ollama.embed_model,
        "gen_model": ollama.gen_model,
    }


@app.post("/doc/insert")
def doc_insert(req: DocInsertRequest):
    global _last_updated
    result = docdb.insert(req.title, req.text)
    _last_updated = time.time()
    return result


@app.get("/doc/list")
def doc_list():
    return {"documents": docdb.list_docs()}


@app.delete("/doc/delete/{doc_id}")
def doc_delete(doc_id: str):
    global _last_updated
    if not docdb.delete_doc(doc_id):
        raise HTTPException(404, "Document not found")
    _last_updated = time.time()
    return {"ok": True}


@app.post("/doc/ask")
def doc_ask(req: DocAskRequest):
    return docdb.ask(req.question, req.k)


@app.get("/last-updated")
def last_updated():
    ago = int(time.time() - _last_updated)
    return {"seconds_ago": ago}
