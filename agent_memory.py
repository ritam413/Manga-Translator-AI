"""
AgentMemory Integration for Manga Translator Full Stack
Provides persistent semantic vector store and O(1) key-value recall
across agent executions and subagent sessions.
"""

import sys
import os
import json
from pathlib import Path

# Ensure agentmemory is imported
try:
    import agentmemory
    import agentmemory.chroma_client
    from agentmemory import (
        create_memory,
        create_unique_memory,
        search_memory,
        get_memories,
        get_memory,
        count_memories,
        wipe_category,
    )

    # Patch ChromaCollectionMemory.query for compatibility with ChromaDB 1.5+ keyword signatures
    def _patched_chroma_query(
        self,
        query_embeddings=None,
        query_texts=None,
        n_results=10,
        where=None,
        where_document=None,
        include=["metadatas", "documents", "distances"],
    ):
        return self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )

    agentmemory.chroma_client.ChromaCollectionMemory.query = _patched_chroma_query

except ImportError:
    print("Error: agentmemory package not installed. Run: pip install agentmemory")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent

# O(1) Fast-Path Key-Value Registry
FAST_RECALL_REGISTRY = {
    "ports": {
        "frontend": 3000,
        "backend": 8000,
        "engine": 8001
    },
    "services": {
        "frontend": "webui/frontend (Next.js 15, TypeScript, Tailwind)",
        "backend": "webui/backend (FastAPI gateway)",
        "engine": "Manga Translator (FastAPI & CLI ML pipeline)"
    },
    "startup": {
        "quick_start": "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\webui'; .\\start.ps1",
        "option_1_one_click": {
            "name": "1-Click Full Stack (Web UI + Backend + Engine)",
            "command": "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\webui'; .\\start.ps1",
            "url": "http://localhost:3000",
            "time_estimate": "~1 minute"
        },
        "option_2_cli_batch": {
            "name": "CLI Direct Batch Translation (Headless)",
            "steps": [
                "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\Manga Translator'; .\\venv\\Scripts\\Activate.ps1",
                "python main.py plan --project-dir '<path_to_project>'",
                "python main.py execute all --project-dir '<path_to_project>'"
            ],
            "output_dir": "<project_dir>\\output\\rendered\\",
            "time_estimate": "~30 seconds + ML inference"
        },
        "option_3_manual_terminals": {
            "name": "Manual 3-Terminal Launch",
            "terminal_1_engine_8001": "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\Manga Translator'; .\\venv\\Scripts\\Activate.ps1; python engine_api.py",
            "terminal_2_backend_8000": "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\webui\\backend'; .\\.venv\\Scripts\\Activate.ps1; python -m uvicorn main:app --reload --port 8000",
            "terminal_3_frontend_3000": "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\webui\\frontend'; npm run dev",
            "time_estimate": "~2 minutes"
        },
        "option_4_docker": {
            "name": "Docker Compose",
            "command": "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\webui'; docker compose up --build",
            "url": "http://localhost:3000",
            "time_estimate": "~3-5 minutes"
        }
    },
    "start": "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\webui'; .\\start.ps1",
    "server": "cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\webui'; .\\start.ps1",
    "pipeline_stages": [
        "detect-text (EasyOCR bounding boxes)",
        "ocr (MangaOCR Japanese transcription)",
        "translate (ArgosTranslate offline / online translation)",
        "inpaint (OpenCV background inpainting)",
        "typeset (PIL English text layout & rendering)"
    ],
    "contracts": {
        "manifest": "project_manifest.json (single source of truth for page states, blocks, coordinates)"
    },
    "tracking_rules": [
        "context.md (architecture and constraints)",
        "features_implemented.md (feature status)",
        "tracker.md (agent handoff log)"
    ]
}

# Domain invariants for semantic vector recall
DEFAULT_INVARIANTS = [
    {
        "category": "startup",
        "content": "Startup Option 1 (1-Click): Run `cd 'e:\\u torrentz\\MangaTranslator FUll STACK\\webui'; .\\start.ps1` to launch Engine (:8001), Backend (:8000), and Frontend (:3000).",
        "metadata": {"type": "startup", "source": "README.md"}
    },
    {
        "category": "startup",
        "content": "Startup Option 2 (CLI Pipeline): Activate `Manga Translator\\venv`, run `python main.py plan --project-dir <dir>`, then `python main.py execute all --project-dir <dir>`.",
        "metadata": {"type": "startup", "source": "README.md"}
    },
    {
        "category": "startup",
        "content": "Startup Option 3 (Manual 3-Terminal): Terminal 1 `python engine_api.py` (:8001), Terminal 2 `uvicorn main:app --port 8000` (:8000), Terminal 3 `npm run dev` (:3000).",
        "metadata": {"type": "startup", "source": "README.md"}
    },
    {
        "category": "startup",
        "content": "Startup Option 4 (Docker): Run `cd webui; docker compose up --build` for containerized frontend, backend, and GPU engine.",
        "metadata": {"type": "startup", "source": "README.md"}
    },
    {
        "category": "architecture",
        "content": "The platform comprises 3 communicating services: Frontend (Next.js :3000), Backend API Gateway (FastAPI :8000), and Translation Engine (FastAPI :8001).",
        "metadata": {"type": "architecture", "source": "context.md"}
    },
    {
        "category": "pipeline_stages",
        "content": "Pipeline stages in order: 1. detect-text (EasyOCR), 2. ocr (MangaOCR), 3. translate (ArgosTranslate offline / online APIs), 4. inpaint (OpenCV / background cleaner), 5. typeset (PIL / font wrapping).",
        "metadata": {"type": "pipeline", "source": "context.md"}
    },
    {
        "category": "ports",
        "content": "Service Port Allocations: Frontend UI on port 3000, Backend API Gateway on port 8000, Translation Engine API on port 8001.",
        "metadata": {"type": "networking", "source": "context.md"}
    },
    {
        "category": "contracts",
        "content": "project_manifest.json is the single source of truth for page states, coordinates, OCR text, and translations across all stages.",
        "metadata": {"type": "data_contract", "source": "context.md"}
    },
    {
        "category": "rules",
        "content": "Every meaningful change must update context.md, features_implemented.md, and tracker.md per project memory rules.",
        "metadata": {"type": "rules", "source": "guidelines.md"}
    },
    {
        "category": "exclusions",
        "content": "Repomix and git ignore heavy artifacts: *.pkl, *.pt, *.onnx, *.bin, training_data/, input/, output/, .hf_cache, .easyocr_cache, .argos_cache, node_modules, .next, .venv.",
        "metadata": {"type": "configuration", "source": "repomix.config.json"}
    }
]


def fast_recall(key: str):
    """O(1) instantaneous lookup from precomputed key-value registry."""
    key_lower = key.strip().lower()
    if key_lower in FAST_RECALL_REGISTRY:
        return FAST_RECALL_REGISTRY[key_lower]
    for k, v in FAST_RECALL_REGISTRY.items():
        if key_lower in k:
            return v
    return None


def seed_memory():
    """Populates agentmemory with system invariants and architectural rules."""
    print("Seeding agentmemory with project knowledge and invariants...")
    
    for item in DEFAULT_INVARIANTS:
        create_unique_memory(
            category=item["category"],
            content=item["content"],
            metadata=item["metadata"]
        )
    
    # Ingest context.md if present
    context_file = PROJECT_ROOT / "context.md"
    if context_file.exists():
        content = context_file.read_text(encoding="utf-8")
        create_unique_memory(
            category="context_doc",
            content=content[:2000],  # index high-level architecture
            metadata={"source": "context.md", "type": "overview"}
        )
        
    print("Successfully seeded agentmemory vector store.")


def query_memory(search_term: str, category: str = None, n_results: int = 5):
    """Performs semantic search across stored memories."""
    # Check O(1) registry first
    instant = fast_recall(search_term)
    if instant is not None:
        return [{"source": "O(1) fast_recall", "content": instant}]
    
    categories = [category] if category else ["architecture", "pipeline_stages", "ports", "contracts", "rules", "exclusions", "context_doc"]
    all_results = []
    for cat in categories:
        try:
            res = search_memory(category=cat, search_text=search_term, n_results=n_results)
            if res:
                all_results.extend(res)
        except Exception as e:
            continue
    return all_results


def list_stats():
    """Returns memory statistics per category."""
    categories = ["architecture", "pipeline_stages", "ports", "contracts", "rules", "exclusions", "context_doc"]
    stats = {}
    for cat in categories:
        count = count_memories(category=cat)
        stats[cat] = count
    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python agent_memory.py seed                  # Seed invariants into ChromaDB")
        print("  python agent_memory.py search <query> [cat]  # Semantic / O(1) recall")
        print("  python agent_memory.py get <key>             # Instant O(1) key-value lookup")
        print("  python agent_memory.py stats                 # View category counts")
        return

    cmd = sys.argv[1]
    if cmd == "seed":
        seed_memory()
    elif cmd == "get":
        if len(sys.argv) < 3:
            print("Please provide a key (e.g. ports, services, pipeline_stages, contracts, tracking_rules)")
            return
        key = sys.argv[2]
        res = fast_recall(key)
        if res is not None:
            print(f"\n[O(1) Instant Recall: '{key}']")
            print(json.dumps(res, indent=2))
        else:
            print(f"Key '{key}' not found in O(1) registry. Available keys: {list(FAST_RECALL_REGISTRY.keys())}")
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Please provide a search query.")
            return
        query = sys.argv[2]
        cat = sys.argv[3] if len(sys.argv) > 3 else None
        results = query_memory(query, category=cat)
        print(f"\n--- Recall results for '{query}' (category={cat}) ---")
        for idx, res in enumerate(results):
            if isinstance(res, dict) and "source" in res and res["source"] == "O(1) fast_recall":
                print(f"[O(1) Instant Cache Hit]")
                print(json.dumps(res["content"], indent=2))
            else:
                text = res.get("document", res.get("text", str(res)))
                meta = res.get("metadata", {})
                print(f"[{idx + 1}] {text}")
                if meta:
                    print(f"    Metadata: {meta}")
    elif cmd == "stats":
        stats = list_stats()
        print("\nAgentMemory Category Counts:")
        for cat, count in stats.items():
            print(f"  - {cat}: {count} items")
        print(f"  - O(1) Fast Keys: {list(FAST_RECALL_REGISTRY.keys())}")
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
