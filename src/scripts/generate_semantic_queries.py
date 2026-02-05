#!/usr/bin/env python3
"""
Generate semantically variable queries for ToolBench tasks using LLM.

This script generates diverse, semantically variable queries for training and
testing by using an LLM to create variations based on task name and ground
truth tools.

Usage:
    python scripts/generate_semantic_queries.py
    (reads config from config/config_query_generation.yaml)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List

from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

# Allow running this file directly without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils import init_llm, load_config, load_models


class QueryBatch(BaseModel):
    """Pydantic model for structured query batch output."""

    queries: List[str] = Field(
        description="List of semantically diverse queries"
    )


def generate_queries_batch(
    llm: Any,
    task_name: str,
    tools: List[str],
    num_queries: int = 10
) -> List[str]:
    """
    Generate a batch of semantically variable queries using LLM.

    Args:
        llm: Initialized LLM model
        task_name: Name of the task
        tools: List of ground truth tool names
        num_queries: Number of queries to generate in this batch

    Returns:
        List of generated query strings
    """
    tools_str = "\n".join([f"- {tool}" for tool in tools])

    prompt = f"""You are generating semantically diverse queries for a task
called "{task_name}".

Task Description: {task_name}

Available Tools (ground truth):
{tools_str}

Generate {num_queries} semantically diverse and variable queries for this task.
Each query should:
1. Be natural and realistic user requests
2. Vary in phrasing, complexity, and specific use cases
3. Cover different aspects and scenarios related to the task
4. Be distinct from each other (avoid repetition)
"""

    structured_llm = llm.with_structured_output(QueryBatch)
    response = structured_llm.invoke(prompt)
    return response.queries


def generate_all_queries(
    llm: Any,
    task_name: str,
    tools: List[str],
    target_train: int = 100,
    target_test: int = 50,
    queries_per_batch: int = 10
) -> tuple[List[str], List[str]]:
    """
    Generate all queries until target_train + target_test, then split.

    Args:
        llm: Initialized LLM model
        task_name: Name of the task
        tools: List of ground truth tool names
        target_train: Target number of train queries (default 100)
        target_test: Target number of test queries (default 50)
        queries_per_batch: Number of queries to generate per batch (default 10)

    Returns:
        Tuple of (train_queries, test_queries)
    """
    total_target = target_train + target_test
    all_queries = []
    seen_queries = set()

    logger.info(
        f"Generating {total_target} total queries "
        f"({target_train} train + {target_test} test)..."
    )
    with tqdm(total=total_target, desc="Total queries") as pbar:
        while len(all_queries) < total_target:
            batch_queries = generate_queries_batch(
                llm=llm,
                task_name=task_name,
                tools=tools,
                num_queries=queries_per_batch
            )

            # Filter out duplicates
            new_queries = []
            for query in batch_queries:
                query_lower = query.lower().strip()
                if query_lower not in seen_queries:
                    seen_queries.add(query_lower)
                    new_queries.append(query)
                    all_queries.append(query)
                    pbar.update(1)
                    if len(all_queries) >= total_target:
                        break

            if new_queries:
                logger.debug(
                    f"Added {len(new_queries)} new queries "
                    f"({len(batch_queries) - len(new_queries)} duplicates skipped)"
                )

    # Split into train and test
    train_queries = all_queries[:target_train]
    test_queries = all_queries[target_train:target_train + target_test]

    logger.info(
        f"Generated {len(train_queries)} train queries and "
        f"{len(test_queries)} test queries (total: {len(all_queries)})"
    )

    return train_queries, test_queries


def update_task_file(
    task_path: Path,
    train_queries: List[str],
    test_queries: List[str],
    backup: bool = True
) -> None:
    """
    Update task JSON file with new queries.
    
    Args:
        task_path: Path to task JSON file
        train_queries: New train queries
        test_queries: New test queries
        backup: Whether to create a backup of the original file
    """
    # Load existing task data
    with task_path.open("r", encoding="utf-8") as f:
        task_data = json.load(f)
    
    # Create backup if requested
    if backup:
        backup_path = task_path.with_suffix(".json.backup")
        with backup_path.open("w", encoding="utf-8") as f:
            json.dump(task_data, f, indent=2)
        logger.info(f"Created backup at {backup_path}")
    
    # Update queries
    task_data["train_queries"] = train_queries
    task_data["test_queries"] = test_queries
    
    # Save updated task data
    with task_path.open("w", encoding="utf-8") as f:
        json.dump(task_data, f, indent=2)
    
    logger.info(f"Updated {task_path} with new queries")


def process_task(
    task_path: Path,
    llm: Any,
    target_train: int = 100,
    target_test: int = 50,
    queries_per_batch: int = 10,
    backup: bool = True
) -> None:
    """
    Process a single task file.

    Args:
        task_path: Path to task JSON file
        llm: Initialized LLM model
        target_train: Target number of train queries
        target_test: Target number of test queries
        queries_per_batch: Number of queries per batch
        backup: Whether to backup original file
    """
    logger.info(f"Processing task: {task_path}")

    # Load task data
    with task_path.open("r", encoding="utf-8") as f:
        task_data = json.load(f)

    task_name = task_data.get("task_name")
    tools = task_data.get("tools", [])

    if not task_name:
        raise ValueError(f"Task file {task_path} missing 'task_name'")
    if not tools:
        raise ValueError(f"Task file {task_path} missing 'tools'")

    logger.info(f"Task: {task_name}")
    logger.info(f"Tools: {len(tools)} tools")

    # Generate queries
    train_queries, test_queries = generate_all_queries(
        llm=llm,
        task_name=task_name,
        tools=tools,
        target_train=target_train,
        target_test=target_test,
        queries_per_batch=queries_per_batch
    )

    # Update task file
    update_task_file(
        task_path=task_path,
        train_queries=train_queries,
        test_queries=test_queries,
        backup=backup
    )


def main():
    """Main entry point."""
    # Load config and models (single config has query_generation section)
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "config.yaml"
    models_path = repo_root / "config" / "models.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Please create config/config.yaml"
        )
    if not models_path.exists():
        raise FileNotFoundError(
            f"Models file not found: {models_path}. "
            "Please create config/models.yaml"
        )
    cfg = load_config(config_path)
    models_cfg = load_models(models_path)
    full_cfg = {**cfg, **models_cfg}
    query_gen_cfg = cfg.get("query_generation", {})

    # Get parameters from config
    target_train = query_gen_cfg.get("target_train", 100)
    target_test = query_gen_cfg.get("target_test", 50)
    queries_per_batch = query_gen_cfg.get("queries_per_batch", 10)
    llm_model_name = query_gen_cfg.get("llm_model", "gpt-4o-mini")
    tasks_directory = query_gen_cfg.get(
        "tasks_directory", "./data/ToolBench/tasks"
    )
    task_file = query_gen_cfg.get("task_file", None)
    backup = query_gen_cfg.get("backup", True)

    logger.info(f"Using LLM model: {llm_model_name}")
    logger.info(f"Target train: {target_train}, Target test: {target_test}")
    logger.info(f"Queries per batch: {queries_per_batch}")

    # Initialize LLM
    llm = init_llm(full_cfg, model_name=llm_model_name)

    # Process tasks
    if task_file:
        # Process single task file
        task_path = Path(task_file).resolve()
        if not task_path.exists():
            raise FileNotFoundError(f"Task file not found: {task_path}")

        process_task(
            task_path=task_path,
            llm=llm,
            target_train=target_train,
            target_test=target_test,
            queries_per_batch=queries_per_batch,
            backup=backup
        )
    else:
        # Process all tasks in directory
        base_path = Path(__file__).resolve().parents[1]
        tasks_dir = (base_path / tasks_directory).resolve()
        if not tasks_dir.exists():
            raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

        task_files = sorted(tasks_dir.glob("*.json"))
        logger.info(f"Found {len(task_files)} task files")

        for task_file_path in task_files:
            try:
                process_task(
                    task_path=task_file_path,
                    llm=llm,
                    target_train=target_train,
                    target_test=target_test,
                    queries_per_batch=queries_per_batch,
                    backup=backup
                )
            except Exception as e:
                logger.error(f"Error processing {task_file_path}: {e}")
                continue

    logger.info("Query generation complete!")


if __name__ == "__main__":
    main()
