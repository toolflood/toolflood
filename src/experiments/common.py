"""
Shared experiment utilities for ToolFlood, RSI, and PoisonRAG runners.

Provides: merge_tools, evaluate_queries, limit_queries, load_existing_results,
get_completed_combinations, write_results_to_disk.
"""  # noqa: E501

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from src.agent import VictimAgent

from src.utils import Tool


def merge_tools(
    benign_tools: List[Tool],
    attacker_tools: List[Tool],
    output_path: Path,
    *,
    attacker_label: str = "attacker",
) -> Tuple[List[Tool], Set[str]]:
    """Merge benign and generated tools, save to JSON; return merged tools + names."""
    logger.info(f"Merging benign and {attacker_label} tools...")

    merged_dict: Dict[str, str] = {tool.name: tool.description for tool in benign_tools}
    attacker_names: Set[str] = set()

    for tool in attacker_tools:
        original_name = tool.name
        tool_name = original_name

        if tool_name in merged_dict:
            suffix = 1
            while f"{original_name}_{suffix}" in merged_dict:
                suffix += 1
            tool_name = f"{original_name}_{suffix}"
            # logger.warning(
            #     f"Generated tool '{original_name}' conflicts with existing name, "
            #     f"renaming to '{tool_name}'"
            # )

        merged_dict[tool_name] = tool.description
        attacker_names.add(tool_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_dict, f, indent=2)

    merged_tools = [
        Tool(tool_id=name, name=name, description=description)
        for name, description in merged_dict.items()
    ]

    total = len(merged_dict)
    logger.success(
        f"Merged {len(benign_tools)} benign + {len(attacker_tools)} {attacker_label} "
        f"= {total} total tools"
    )

    return merged_tools, attacker_names


async def evaluate_queries(
    agent: "VictimAgent",
    queries: Sequence[str],
    attacker_tool_names: Set[str],
) -> Dict[str, Any]:
    """
    Evaluate victim agent on a list of queries.

    Returns a dict with total_queries, successful_selections, attacker_selections,
    benign_selections, no_selection, top_k_tools, query_results.
    """  # noqa: E501
    logger.info(f"Evaluating {len(queries)} queries...")

    results: Dict[str, Any] = {
        "total_queries": len(queries),
        "successful_selections": 0,
        "attacker_selections": 0,
        "benign_selections": 0,
        "no_selection": 0,
        "top_k_tools": [],
        "query_results": [],
    }

    for i, query in enumerate(
        tqdm(queries, desc="Evaluating queries", total=len(queries))
    ):
        try:
            top_k_tools = await agent.retrieve_topk(query)
            results["top_k_tools"].append([t.name for t in top_k_tools])

            selected_tool = await agent.llm_choose_tool(query, top_k_tools)

            if selected_tool is None:
                results["no_selection"] += 1
                results["query_results"].append({
                    "query": query,
                    "selected_tool": None,
                    "is_attacker": False,
                })
                continue

            results["successful_selections"] += 1
            is_attacker = selected_tool.name in attacker_tool_names
            if is_attacker:
                results["attacker_selections"] += 1
            else:
                results["benign_selections"] += 1

            results["query_results"].append({
                "query": query,
                "selected_tool": {
                    "tool_id": selected_tool.tool_id,
                    "name": selected_tool.name,
                    "description": selected_tool.description,
                },
                "is_attacker": is_attacker,
            })
        except Exception as exc:
            logger.error(f"Error processing query {i + 1}: {exc}")
            results["no_selection"] += 1
            results["top_k_tools"].append([])
            results["query_results"].append({
                "query": query,
                "selected_tool": None,
                "is_attacker": False,
                "error": str(exc),
            })

    return results


def limit_queries(
    train_queries: List[str],
    test_queries: List[str],
    max_train: Optional[int],
    max_test: Optional[int],
) -> Tuple[List[str], List[str]]:
    """Optionally shuffle and slice train/test query lists by limits."""
    if max_train is not None:
        random.shuffle(train_queries)
        train_queries = train_queries[:max_train]
    if max_test is not None:
        random.shuffle(test_queries)
        test_queries = test_queries[:max_test]
    return train_queries, test_queries


def load_existing_results(results_json_path: Path) -> List[Dict[str, Any]]:
    """Load existing results from JSON file if it exists. Returns [] on missing or error."""
    if not results_json_path.exists():
        return []
    try:
        with results_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info(
                f"Loaded {len(data)} existing results from {results_json_path}"
            )
            return data
        logger.warning(
            f"Unexpected JSON shape in {results_json_path}; expected a list."
        )
        return []
    except Exception as exc:
        logger.warning(f"Failed to load {results_json_path}: {exc}")
        return []


def get_completed_combinations(
    results: List[Dict[str, Any]],
    keys: Sequence[str],
) -> Set[Tuple[Any, ...]]:
    """
    Return set of tuple keys for completed experiment combinations.

    Each result becomes (result[k] for k in keys). Results with any falsy
    value for those keys are skipped.
    """
    completed: Set[Tuple[Any, ...]] = set()
    for r in results:
        t = tuple(r.get(k, "") for k in keys)
        if all(t):
            completed.add(t)
    return completed


def write_results_to_disk(
    all_results: List[Dict[str, Any]],
    results_json_path: Path,
    table_df: pd.DataFrame,
    results_table_path: Path,
) -> None:
    """Write full results JSON and results table CSV."""
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.success(f"Results saved to: {results_json_path}")
    table_df.to_csv(results_table_path, index=False)
    logger.success(f"Results table updated: {results_table_path}")  # noqa: E501
