#!/usr/bin/env python3
"""
Experiment script for Random-Sybil Injection (RSI) evaluation.

This script:
1. Generates random tools using GPT (no query conditioning).
2. Merges random and benign tools.
3. Builds a new FAISS vector store on the merged tool set.
4. Evaluates victim agents on held-out test queries.
5. Reports ASR, TDR, and Mean Domination metrics.

CLI:
    python -m rsi.run_experiment --config config/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

# Allow running this file directly without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.common import (
    evaluate_queries,
    get_completed_combinations,
    limit_queries,
    load_existing_results,
    merge_tools,
    write_results_to_disk,
)
from src.agent import VictimAgent
from src.metrics import (
    calculate_asr,
    calculate_mean_domination,
    calculate_tdr,
)
from src.utils import (
    Tool,
    get_base_path,
    init_embedding_model,
    init_llm,
    load_config,
    load_experiment_config,
    load_models,
    load_queries_from_tasks,
    load_tools,
    load_vector_store,
    resolve_path,
)
from src.scripts.build_vectorstore import init_vector_store

from src.attacks.rsi_attack import (
    RSIAttack,
    RSIConfig,
    load_rsi_attack_config,
)


def update_results_table(
    all_results: List[Dict[str, Any]],
    exp_cfg,
) -> pd.DataFrame:
    """Create results table similar to ToolFlood's runner (Test split only)."""
    benchmark_name = Path(exp_cfg.benign_data_directory).name
    rows: List[Dict[str, Any]] = []
    for r in all_results:
        test_stats = r.get("test_stats", {})
        rows.append(
            {
                "Split": "Test",
                "Model": r.get("model", "N/A"),
                "Victim_Embedding_Model": r.get(
                    "victim_embedding_model", "N/A"
                ),
                "Benchmark": benchmark_name,
                "Scenario": r.get("domain", "N/A"),
                "Num_Injected_Tools": r.get("num_injected_tools", 0),
                "Benign_Tools": r.get("num_benign_tools", 0),
                "Total_Queries": test_stats.get("total_queries", 0),
                "ASR": f"{test_stats.get('asr', 0.0):.4f}",
                "TDR": f"{test_stats.get('tdr', 0.0):.4f}",
                "Mean_Domination": f"{test_stats.get('mean_domination', 0.0):.4f}",  # noqa: E501
                "Top_K": r.get("top_k", None),
            }
        )
    return pd.DataFrame(rows)


def save_results(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    exp_cfg,
    results_json_path: Path,
    results_table_path: Path,
) -> None:
    """Save aggregated results JSON and CSV."""
    results_table = update_results_table(all_results, exp_cfg)
    write_results_to_disk(
        all_results, results_json_path, results_table, results_table_path
    )


async def _run_experiment_for_task(
    *,
    task_name: Optional[str],
    task_str: str,
    exp_cfg,
    benign_data_dir: Path,
    output_dir: Path,
    merged_tools: List[Tool],
    attacker_names: Set[str],
    merged_vectorstore: Any,
    victim_embedding_model: Any,
    victim_llm: Any,
    victim_embedding_model_name: str,
    victim_model_name: str,
) -> Dict[str, Any]:
    """Run RSI evaluation for one (task, victim_emb, victim_model)."""
    logger.info(
        f"Running RSI: task={task_str}, model={victim_model_name}, "
        f"victim_emb={victim_embedding_model_name}"
    )

    # Output directory structure matches ToolFlood runner:
    # output_dir/<task>/<victim_emb_...>/
    task_dir = output_dir / task_str if task_str != "all" else output_dir
    embedding_subdir = f"victim_emb_{victim_embedding_model_name}"
    run_dir = task_dir / embedding_subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load test queries for this task (RSI doesn't use train queries)
    tasks_path = benign_data_dir / "tasks"
    task_list = [task_name] if task_name else None
    _, test_queries = load_queries_from_tasks(
        tasks_path, task_names=task_list
    )
    _, test_queries = limit_queries(
        [],
        test_queries,
        None,
        exp_cfg.max_test_queries,
    )
    logger.info(
        f"Task {task_str}: {len(test_queries)} test queries"
    )

    top_k = exp_cfg.max_train_evaluation_queries or 5
    agent = VictimAgent(
        tools=merged_tools,
        vectorstore=merged_vectorstore,
        embedding_model=victim_embedding_model,
        llm=victim_llm,
        top_k=top_k,
        verbose=True,
    )

    eval_results = await evaluate_queries(agent, test_queries, attacker_names)

    asr = calculate_asr(
        eval_results["attacker_selections"],
        eval_results["benign_selections"],
        eval_results["no_selection"],
    )
    tdr = calculate_tdr(eval_results["top_k_tools"], attacker_names, k=top_k)
    mean_dom = calculate_mean_domination(
        eval_results["top_k_tools"], attacker_names, k=top_k
    )

    result: Dict[str, Any] = {
        "domain": task_str,
        "model": victim_model_name,
        "victim_embedding_model": victim_embedding_model_name,
        "top_k": top_k,
        "num_injected_tools": len(attacker_names),
        "num_benign_tools": len(merged_tools) - len(attacker_names),
        "test_stats": {
            "total_queries": eval_results["total_queries"],
            "successful_selections": eval_results["successful_selections"],
            "attacker_selections": eval_results["attacker_selections"],
            "benign_selections": eval_results["benign_selections"],
            "no_selection": eval_results["no_selection"],
            "asr": asr,
            "tdr": tdr,
            "mean_domination": mean_dom,
        },
        "test_query_results": eval_results["query_results"],
    }

    # Per-run summary CSV in the folder
    per_run_df = update_results_table([result], exp_cfg)
    per_run_df.to_csv(run_dir / "results_table.csv", index=False)

    # Keep a metrics.json for convenience (compatible with prior output)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": task_str,
                "model": victim_model_name,
                "victim_embedding_model": victim_embedding_model_name,
                "asr": asr,
                "tdr": tdr,
                "mean_domination": mean_dom,
                "k": top_k,
                "total_queries": eval_results["total_queries"],
                "successful_selections": eval_results[
                    "successful_selections"
                ],
                "attacker_selections": eval_results["attacker_selections"],
                "benign_selections": eval_results["benign_selections"],
                "no_selection": eval_results["no_selection"],
            },
            f,
            indent=2,
        )

    logger.success(
        f"Completed RSI: task={task_str}, model={victim_model_name}, "
        f"victim_emb={victim_embedding_model_name}"
    )
    return result


async def _run_single_experiment(cfg_path: Path, models_path: Path) -> int:
    """Run RSI experiments from config and models files, organized by task."""
    cfg = load_config(cfg_path)
    models_cfg = load_models(models_path)
    full_cfg = {**cfg, **models_cfg}
    exp_cfg = load_experiment_config(cfg_path)
    rsi_cfg = load_rsi_attack_config(cfg_path)

    base_path = get_base_path(cfg_path)
    benign_data_dir = resolve_path(
        base_path,
        rsi_cfg.benign_data_directory or exp_cfg.benign_data_directory,
    )
    output_dir = resolve_path(base_path, rsi_cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Benign data directory: {benign_data_dir}")
    logger.info(f"Output directory: {output_dir}")

    results_json_path = output_dir / "results_full.json"
    results_table_path = output_dir / "results_table.csv"

    if exp_cfg.hard_reset:
        logger.info("Hard reset enabled: starting fresh")
        all_results: List[Dict[str, Any]] = []
    else:
        all_results = load_existing_results(results_json_path)

    completed = get_completed_combinations(
        all_results, ["domain", "model", "victim_embedding_model"]
    )
    if completed:
        logger.info(
            f"Found {len(completed)} completed combinations; "
            "will skip them."
        )

    # Load benign tools
    benign_tools = load_tools(benign_data_dir / "tools.json")
    logger.info(f"Loaded {len(benign_tools)} benign tools")

    # Initialize LLM generator for attack (required)
    generator_model_name = rsi_cfg.generator_model
    if not generator_model_name:
        # Default to victim model if not specified
        generator_model_name = (
            exp_cfg.victim_models[0] if exp_cfg.victim_models else None
        )
        if not generator_model_name:
            raise ValueError(
                "generator_model must be specified in rsi_attack config"
            )
        logger.info(
            "No generator_model specified, using first victim model: "
            f"{generator_model_name}"
        )
    llm_generator = init_llm(full_cfg, model_name=generator_model_name)
    logger.info(f"Using LLM generator: {generator_model_name}")

    # Determine tasks to process
    tasks_path = benign_data_dir / "tasks"
    if exp_cfg.task_names:
        # Use specified task names
        tasks = exp_cfg.task_names
    else:
        # Load all tasks from the tasks directory
        task_files = sorted(tasks_path.glob("*.json"))
        tasks = []
        for task_file in task_files:
            task_data = json.loads(task_file.read_text(encoding="utf-8"))
            task_name = task_data.get("task_name")
            if task_name:
                tasks.append(task_name)
        if not tasks:
            # Fallback: use None to process all tasks together
            tasks = [None]

    logger.info(f"Processing {len(tasks)} task(s)")

    # Generate attacker tools once (shared across all tasks)
    logger.info("Generating attacker tools (shared across all tasks)...")
    rsi_config = RSIConfig(
        num_tools=rsi_cfg.num_tools,
        batch_size=rsi_cfg.batch_size,
        temperature=rsi_cfg.temperature,
    )
    attack = RSIAttack(
        benign_tools,
        rsi_config,
        llm_generator=llm_generator,
    )
    attacker_tools, attack_details = attack.attack()

    # Save attack artifacts once (shared)
    shared_output_dir = output_dir / "shared"
    shared_output_dir.mkdir(parents=True, exist_ok=True)
    with (shared_output_dir / "attack_tools.json").open("w", encoding="utf-8") as f:
        tools_dict = {t.name: t.description for t in attacker_tools}
        json.dump(tools_dict, f, indent=2)
    with (shared_output_dir / "attack_details.json").open("w", encoding="utf-8") as f:
        json.dump(attack_details, f, indent=2)
    logger.success(
        f"Generated {len(attacker_tools)} attacker tools (shared)"
    )

    # Determine grids to process
    victim_embedding_model_names = (
        exp_cfg.victim_embedding_models
        if getattr(exp_cfg, "victim_embedding_models", None)
        else ["text-embedding-3-small"]
    )
    victim_model_names = (
        exp_cfg.victim_models
        if getattr(exp_cfg, "victim_models", None)
        else ["gpt-4o-mini"]
    )

    # Initialize embedding models (cache)
    victim_embedding_models: Dict[str, Any] = {}
    for name in victim_embedding_model_names:
        victim_embedding_models[name] = init_embedding_model(
            full_cfg, model_name=name
        )

    # Build shared vector stores per embedding model
    shared_vectorstores: Dict[str, Tuple[List[Tool], Set[str], Any]] = {}
    for victim_emb_name in victim_embedding_model_names:
        logger.info(
            f"Building shared vector store for embedding: {victim_emb_name}"
        )
        merged_tools, attacker_names = merge_tools(
            benign_tools,
            attacker_tools,
            shared_output_dir / f"merged_tools_{victim_emb_name}.json",
            attacker_label="random",
        )
        vectorstore_path = shared_output_dir / f"vectorstore_{victim_emb_name}"
        init_vector_store(
            merged_tools,
            victim_embedding_models[victim_emb_name],
            vectorstore_path=vectorstore_path,
            force_rebuild=True,
        )
        merged_vectorstore = load_vector_store(
            vectorstore_path, victim_embedding_models[victim_emb_name]
        )
        shared_vectorstores[victim_emb_name] = (
            merged_tools,
            attacker_names,
            merged_vectorstore,
        )
        logger.success(
            f"Built shared vector store for {victim_emb_name}: "
            f"{len(merged_tools)} total tools"
        )

    total_combinations = (
        len(tasks)
        * len(victim_embedding_model_names)
        * len(victim_model_names)
    )
    completed_count = len(completed)
    logger.info(
        f"Total combinations to process: {total_combinations} "
        f"({completed_count} already completed)"
    )

    for task_name in tasks:
        task_str = task_name if task_name else "all"
        for victim_emb_name in victim_embedding_model_names:
            merged_tools, attacker_names, merged_vectorstore = shared_vectorstores[
                victim_emb_name
            ]
            for victim_model_name in victim_model_names:
                key = (task_str, victim_model_name, victim_emb_name)
                if key in completed:
                    logger.info(
                        f"Skipping already completed: {task_str} / "
                        f"{victim_model_name} / victim_emb: {victim_emb_name}"
                    )
                    continue

                victim_llm = init_llm(full_cfg, model_name=victim_model_name)
                result = await _run_experiment_for_task(
                    task_name=task_name,
                    task_str=task_str,
                    exp_cfg=exp_cfg,
                    benign_data_dir=benign_data_dir,
                    output_dir=output_dir,
                    merged_tools=merged_tools,
                    attacker_names=attacker_names,
                    merged_vectorstore=merged_vectorstore,
                    victim_embedding_model=victim_embedding_models[
                        victim_emb_name
                    ],
                    victim_llm=victim_llm,
                    victim_embedding_model_name=victim_emb_name,
                    victim_model_name=victim_model_name,
                )

                all_results.append(result)
                completed.add(key)
                completed_count += 1

                save_results(
                    all_results,
                    output_dir,
                    exp_cfg,
                    results_json_path,
                    results_table_path,
                )
                logger.success(
                    f"Progress: {completed_count}/{total_combinations} completed"
                )

    logger.success("All RSI experiments completed.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run Random-Sybil Injection (RSI) experiment.",
    )
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
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    models_path = args.models.resolve()
    return asyncio.run(_run_single_experiment(cfg_path, models_path))


if __name__ == "__main__":
    raise SystemExit(main())
