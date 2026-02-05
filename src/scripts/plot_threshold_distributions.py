#!/usr/bin/env python3
"""
Plot the distribution of closest thresholds for ToolBench and ToolE.

This script computes the closest benign tool distance (threshold) for each
query in ToolBench and ToolE datasets, and plots their distributions in the
same plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from loguru import logger

# Ensure repo root is on the Python path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (  # noqa: E402
    Tool,
    cosine_distance,
    get_base_path,
    init_embedding_model,
    load_config,
    load_models,
    load_queries_from_tasks,
    load_tools,
)


def compute_query_thresholds(
    queries: list[str],
    benign_vectorstore: FAISS,
    embedding_model,
) -> list[float]:
    """
    Compute per-query thresholds based on closest benign tool distance.

    Args:
        queries: List of query strings
        benign_vectorstore: Pre-built FAISS vector store from benign tools
        embedding_model: Embedding model instance

    Returns:
        List of threshold values (one per query)
    """
    logger.info(f"Computing thresholds for {len(queries)} queries...")

    # Compute thresholds for each query
    query_thresholds = []
    for i, query in enumerate(queries):
        # Retrieve top 1 result (closest benign tool)
        results = benign_vectorstore.similarity_search(query, k=1)
        if results:
            # Get the closest benign tool's description
            closest_doc = results[0]
            closest_tool_description = closest_doc.page_content

            # Calculate cosine distance using same method as ToolFlood attack
            query_emb = np.array(embedding_model.embed_query(query))
            closest_tool_emb = np.array(
                embedding_model.embed_query(closest_tool_description)
            )
            closest_distance = cosine_distance(query_emb, closest_tool_emb)

            query_thresholds.append(closest_distance)
        else:
            # Fallback: use max distance if no results found
            logger.warning(
                f"Query {i+1}: No benign tool found in vector store"
            )
            query_thresholds.append(1.0)  # Max distance as fallback

    return query_thresholds


def compute_all_thresholds(
    data_dir: Path,
    embedding_model,
) -> list[float]:
    """
    Compute thresholds for all queries in a dataset.

    Args:
        data_dir: Path to dataset directory (should contain tasks/ and
            tools.json)
        embedding_model: Embedding model instance

    Returns:
        List of all threshold values across all tasks
    """
    # Load benign tools
    benign_tools = load_tools(data_dir / "tools.json")

    # Build vector store from benign tools (once per dataset)
    logger.info(
        f"Building vector store from {len(benign_tools)} benign tools..."
    )
    benign_documents = [
        Document(
            page_content=tool.description,
            metadata={
                "tool_name": tool.name,
                "description": tool.description
            }
        )
        for tool in benign_tools
    ]
    benign_vectorstore = FAISS.from_documents(
        benign_documents, embedding_model,
        distance_strategy=DistanceStrategy.COSINE
    )

    # Load all queries from all tasks
    train_queries, _ = load_queries_from_tasks(
        data_dir / "tasks",
        task_names=None  # Load all tasks
    )

    # Compute thresholds using the pre-built vector store
    thresholds = compute_query_thresholds(
        train_queries,
        benign_vectorstore,
        embedding_model,
    )

    return thresholds


def print_statistics(dataset_name: str, thresholds: list[float]) -> None:
    """Print min, max, mean, and quantile statistics for thresholds."""
    thresholds_array = np.array(thresholds)
    quantiles = np.percentile(thresholds_array, [5, 10, 25, 50, 75, 90, 95])
    print(f"\n{dataset_name} Statistics:")
    print(f"  Min:   {np.min(thresholds_array):.4f}")
    print(f"  5th:   {quantiles[0]:.4f}")
    print(f"  10th:  {quantiles[1]:.4f}")
    print(f"  25th:  {quantiles[2]:.4f}")
    print(f"  50th:  {quantiles[3]:.4f} (median)")
    print(f"  75th:  {quantiles[4]:.4f}")
    print(f"  90th:  {quantiles[5]:.4f}")
    print(f"  95th:  {quantiles[6]:.4f}")
    print(f"  Max:   {np.max(thresholds_array):.4f}")
    print(f"  Mean:  {np.mean(thresholds_array):.4f}")
    print(f"  Count: {len(thresholds_array)}")


def plot_distributions(
    toolbench_thresholds: list[float],
    toole_thresholds: list[float],
    output_path: Path | None = None,
) -> None:
    """
    Plot distributions of thresholds for both datasets in the same plot.

    Args:
        toolbench_thresholds: List of thresholds for ToolBench
        toole_thresholds: List of thresholds for ToolE
        output_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms with transparency for overlap
    ax.hist(
        toolbench_thresholds,
        bins=50,
        alpha=0.6,
        label="ToolBench",
        color="blue",
        edgecolor="black",
    )
    ax.hist(
        toole_thresholds,
        bins=50,
        alpha=0.6,
        label="ToolE",
        color="red",
        edgecolor="black",
    )

    ax.set_xlabel("Closest Threshold (Cosine Distance)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Distribution of Closest Thresholds\n(ToolBench vs ToolE)",
        fontsize=14,
        fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.success(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main() -> int:
    """Main function to compute and plot threshold distributions."""
    logger.info("Starting threshold distribution analysis...")

    # Load config and models from top-level `config` directory
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    models_path = PROJECT_ROOT / "config" / "models.yaml"
    cfg = load_config(config_path)
    models_cfg = load_models(models_path)
    full_cfg = {**cfg, **models_cfg}
    base_path = get_base_path(config_path)

    # Initialize embedding model (using text-embedding-3-small as default)
    embedding_model = init_embedding_model(
        full_cfg, model_name="text-embedding-3-small"
    )

    # Compute thresholds for ToolBench
    logger.info("Processing ToolBench...")
    toolbench_data_dir = base_path / "data" / "ToolBench"
    toolbench_thresholds = compute_all_thresholds(
        toolbench_data_dir, embedding_model
    )

    # Compute thresholds for ToolE
    logger.info("Processing ToolE...")
    toole_data_dir = base_path / "data" / "ToolE"
    toole_thresholds = compute_all_thresholds(toole_data_dir, embedding_model)

    # Print statistics
    print("=" * 60)
    print("THRESHOLD DISTRIBUTION STATISTICS")
    print("=" * 60)
    print_statistics("ToolBench", toolbench_thresholds)
    print_statistics("MetaTool", toole_thresholds)
    print("=" * 60)

    # Plot distributions
    output_path = Path(__file__).parent / "threshold_distribution_plot.png"
    plot_distributions(toolbench_thresholds, toole_thresholds, output_path)

    logger.success("Analysis complete!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
