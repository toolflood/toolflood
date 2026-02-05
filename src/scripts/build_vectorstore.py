#!/usr/bin/env python3
"""
Build local evaluation data from queries.csv and tools.json, and create
a vector store for embedding-based retrieval using LangChain.

CLI:
  python scripts/build_vectorstore.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from loguru import logger

from src.utils import (
    get_base_path,
    get_data_dir,
    init_embedding_model,
    load_config,
    load_models,
    load_tools,
    resolve_path,
)


def init_vector_store(
    tools: List[dict],
    embedding_model: Any,
    vectorstore_path: Path,
    force_rebuild: bool = False
) -> FAISS:
    """Initialize or load vector store from tools."""
    if not force_rebuild and vectorstore_path.exists():
        try:
            logger.info(
                f"Loading existing vector store from {vectorstore_path}"
            )
            vectorstore = FAISS.load_local(
                str(vectorstore_path),
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.success(
                f"Loaded existing vector store from {vectorstore_path}"
            )
            return vectorstore
        except Exception as e:
            logger.warning(
                f"Failed to load existing vector store: {e}. Rebuilding..."
            )

    logger.info(f"Creating vector store from {len(tools)} tools")
    documents = [
        Document(
            page_content=f"{tool.name}: {tool.description}",
            metadata={
                "tool_name": tool.name,
                "description": tool.description
            }
        )
        for tool in tools
    ]

    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_path))
    logger.success(f"Vector store created -> {vectorstore_path}")
    return vectorstore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config YAML (default: config/config.yaml)",
    )
    ap.add_argument(
        "--models",
        type=Path,
        default=Path("config/models.yaml"),
        help="Path to models YAML (default: config/models.yaml)",
    )
    ap.add_argument(
        "--data-dir",
        help="Override data directory path (relative to config base path or absolute)"
    )
    args = ap.parse_args()

    logger.info(
        f"Starting vector store construction with config: {args.config}"
    )
    cfg_path = args.config.resolve()
    models_path = args.models.resolve()
    cfg = load_config(cfg_path)
    models_cfg = load_models(models_path)
    full_cfg = {**cfg, **models_cfg}

    if args.data_dir:
        base_path = get_base_path(cfg_path)
        data_dir = resolve_path(base_path, args.data_dir)
    else:
        data_dir = get_data_dir(cfg_path, cfg)

    data_dir.mkdir(parents=True, exist_ok=True)

    tools = load_tools(data_dir / "tools.json")
    logger.debug(f"Loaded {len(tools)} tools from {data_dir / 'tools.json'}")
    embedding_model = init_embedding_model(full_cfg)
    init_vector_store(tools, embedding_model, data_dir / "vectorstore")
    logger.success("Vector store construction completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
