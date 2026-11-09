from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn

from citeguard.api import create_app
from citeguard.db import DatabaseManager
from citeguard.grounding import GroundingVerifier
from citeguard.indexer import IncrementalSyncEngine
from citeguard.ollama_client import OllamaChatGenerator, OllamaEmbedder
from citeguard.qdrant_store import QdrantVectorStore
from citeguard.retriever import HybridRRFRetriever
from citeguard.schemas import DocumentSyncRequest, MetadataFilter

app = typer.Typer(
    name="citeguard",
    help="Incremental Qdrant Hybrid RAG and Local Ollama Grounding Verifier CLI",
)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8090,
    db_url: str = "sqlite:///./citeguard.db",
) -> None:
    """Run FastAPI HTTP server."""
    uvicorn.run(create_app(db_url=db_url), host=host, port=port)


@app.command()
def sync(
    doc_key: Annotated[str, typer.Option("--doc-key", help="Document key")],
    title: Annotated[str, typer.Option("--title", help="Document title")],
    url: str = "https://official.example.org/policy",
    file: Annotated[Optional[Path], typer.Option("--file", help="Local Markdown/HTML file path")] = None,
    category: str = "commerce_policy",
    language: str = "ko",
    db_url: str = "sqlite:///./citeguard.db",
    qdrant_url: str = "http://localhost:6333",
    collection: str = "citeguard_chunks",
    ollama_url: str = "http://localhost:11434",
    embed_model: str = "all-minilm",
) -> None:
    """Incrementally sync a policy document into Qdrant."""

    async def _run() -> None:
        db_mgr = DatabaseManager(db_url)
        db_mgr.create_tables()
        embedder = OllamaEmbedder(base_url=ollama_url, model=embed_model)
        qdrant = QdrantVectorStore(base_url=qdrant_url, collection_name=collection)
        engine = IncrementalSyncEngine(db_manager=db_mgr, embedder=embedder, qdrant=qdrant)

        content = file.read_text(encoding="utf-8") if file else None
        res = await engine.sync_document(
            DocumentSyncRequest(
                doc_key=doc_key,
                title=title,
                source_url=url,
                category=category,
                language=language,
                content=content,
            )
        )
        print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))

    asyncio.run(_run())


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to query")],
    category: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = 4,
    db_url: str = "sqlite:///./citeguard.db",
    qdrant_url: str = "http://localhost:6333",
    collection: str = "citeguard_chunks",
    ollama_url: str = "http://localhost:11434",
    embed_model: str = "all-minilm",
    chat_model: str = "qwen2.5:0.5b",
) -> None:
    """Query Hybrid RAG and verify local LLM answer."""

    async def _run() -> None:
        db_mgr = DatabaseManager(db_url)
        db_mgr.create_tables()
        embedder = OllamaEmbedder(base_url=ollama_url, model=embed_model)
        generator = OllamaChatGenerator(base_url=ollama_url, model=chat_model)
        qdrant = QdrantVectorStore(base_url=qdrant_url, collection_name=collection)
        retriever = HybridRRFRetriever(db_manager=db_mgr, embedder=embedder, qdrant=qdrant)
        verifier = GroundingVerifier(db_manager=db_mgr)

        filters = MetadataFilter(category=category, language=language)
        hits = await retriever.retrieve(query=question, filters=filters, top_k=top_k)
        resp = await verifier.generate_and_verify(
            question=question,
            retrieved_chunks=hits,
            generator=generator,
        )
        print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))

    asyncio.run(_run())


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv, standalone_mode=False)
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
