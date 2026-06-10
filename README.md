# VectorDB v0.1.0

A local vector database explorer with a FastAPI backend and a single-file dashboard. Search vectors with HNSW, KD-Tree, or brute force, visualize clusters in 2D/3D PCA, benchmark algorithms, and run document RAG with Ollama.

**Author & contributor:** [Iconic Venom](https://github.com/iconicvenom)

---

## Features

- **Semantic search** — query demo vectors with cosine, euclidean, or manhattan distance
- **Three index algorithms** — HNSW, KD-Tree, brute force with live timing comparison
- **PCA visualization** — 2D Chart.js scatter and 3D Three.js views with cluster edges and hub
- **Document RAG** — chunk, embed, and chat over your own text (requires Ollama)
- **Live stats** — vector count, dimensions, memory, HNSW index size, uptime

The dashboard ships with **40 demo vectors** across CS, Math, Food, and Sports categories. Search and visualization work without Ollama; embeddings and RAG need it.

---

## Quick start (Docker)

Recommended way to run the app.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- (Optional) [Ollama](https://ollama.com/) on your host for embeddings & RAG

### Run

From the project root:

```bash
git clone https://github.com/iconicvenom/vectordb.git
cd vectordb
docker compose up --build
```

Open **[http://localhost:8080](http://localhost:8080)**.

Stop with `Ctrl+C`, or run detached:

```bash
docker compose up --build -d
docker compose logs -f
docker compose down
```

### Ollama with Docker

The container talks to Ollama on your **host** via `host.docker.internal`. On your machine:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
```

If Ollama is offline, the dashboard still works with demo data; the header shows an Ollama status warning.

### Build & run without Compose

```bash
cd vectordb
docker build -t vectordb .
docker run --rm -p 8080:8080 \
  -e OLLAMA_BASE=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  vectordb
```

---

## Quick start (local Python)

```bash
cd vectordb
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8080 --reload
```

Open **[http://localhost:8080](http://localhost:8080)**.

---

## How to use the dashboard

### 1. Dashboard (home)

- **Semantic Search** — enter a query (e.g. `sorting algorithm`), set Top K, choose distance metric, click **Search**. Results appear in the table; matching points highlight on the scatter plots.
- **Compare All Algorithms** — runs HNSW, KD-Tree, and brute force on the same query and shows latency.
- **Upload Document** — paste text, click **Embed & Insert** (needs Ollama). Enables RAG.
- **2D / 3D PCA** — explore vector clusters; hover for labels; use options menu to reset highlights or switch views.
- **RAG Chat** — ask questions about uploaded documents (needs embedded docs + Ollama).

### 2. Sidebar navigation

| Page | Purpose |
|------|---------|
| Dashboard | Search, charts, upload, RAG |
| Search | Shortcut to dashboard search |
| Compare | Algorithm benchmark info |
| Documents | List embedded documents |
| RAG Chat | Full-page chat over your docs |
| Collections / API Keys / Settings | Collection and server info |

### 3. Status bar (footer)

Shows collection name, vector count, dimensions, index params, and last updated time. Use **↻** to refresh stats and reload PCA data.

---

## API reference

Base URL: `http://localhost:8080`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/search` | Semantic search (`q`, `k`, `metric`, `algo`) |
| POST | `/insert` | Insert a vector |
| DELETE | `/delete/{item_id}` | Delete a vector |
| GET | `/items` | List all vectors |
| GET | `/benchmark` | Compare algorithm timings |
| GET | `/hnsw-info` | HNSW index parameters |
| GET | `/stats` | System statistics |
| GET | `/pca-coords` | PCA coordinates and cluster edges |
| GET | `/status` | Server and Ollama status |
| GET | `/models` | Available Ollama models |
| POST | `/doc/insert` | Embed and store a document |
| GET | `/doc/list` | List documents |
| DELETE | `/doc/delete/{doc_id}` | Remove a document |
| POST | `/doc/ask` | RAG question over documents |
| GET | `/last-updated` | Seconds since last data change |

Example:

```bash
curl "http://localhost:8080/search?q=binary+tree&k=5&metric=cosine&algo=hnsw"
```

Interactive docs: **[http://localhost:8080/docs](http://localhost:8080/docs)**

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP port (Docker) |
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama API URL |

---

## Project structure

```
vector-db/
├── README.md
├── docker-compose.yml
├── .gitignore
├── .dockerignore
└── vectordb/
    ├── Dockerfile
    ├── main.py          # FastAPI backend
    ├── index.html       # Dashboard (HTML + CSS + JS)
    └── requirements.txt
```

---

## Stack

- **Backend:** FastAPI, NumPy, SciPy, hnswlib
- **Frontend:** Chart.js, Three.js (CDN), no build step
- **Embeddings / LLM:** Ollama (`nomic-embed-text`, `llama3.2`)

---

## Push to GitHub

If you are forking or republishing this project:

```bash
git init
git add .
git commit -m "Add VectorDB with Docker support"
git branch -M main
git remote add origin https://github.com/iconicvenom/vectordb.git
git push -u origin main
```

Create the empty repo on GitHub first (**New repository** → `vectordb`), or use the GitHub CLI:

```bash
gh auth login
gh repo create iconicvenom/vectordb --public --source=. --remote=origin --push
```

---

## License

MIT — see repository license file if present.

---

## Author

**Iconic Venom** — author & contributor  
GitHub: [@iconicvenom](https://github.com/iconicvenom)
