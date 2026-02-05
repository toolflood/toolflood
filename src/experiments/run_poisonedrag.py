#!/usr/bin/env python3
"""
Experiment script for black-box PoisonRAG evaluation.

This script mirrors the high-level ToolFlood evaluation pipeline, but uses a
much simpler PoisonRAG-style attack:

1. Generates poisoned tools from training queries (no optimizer LLM).
2. Merges poisoned and benign tools.
3. Builds a new FAISS vector store on the merged tool set.
4. Evaluates victim agents on held-out test queries.
5. Reports ASR, TDR, and Mean Domination metrics.

CLI:
    python -m poisonrag.run_experiment --config config/config.yaml
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
    load_agent_config,
    load_config,
    load_experiment_config,
    load_models,
    load_queries_from_tasks,
    load_tools,
    load_vector_store,
    resolve_path,
)
from src.scripts.build_vectorstore import init_vector_store

from src.attacks.poisonedrag_attack import (
    PoisonRAGBlackBoxAttack,
    PoisonRAGConfig,
    load_poisonrag_config,
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
                "Attack_Embedding_Model": r.get("attack_embedding_model", "N/A"),
                "Victim_Embedding_Model": r.get("victim_embedding_model", "N/A"),
                "Benchmark": benchmark_name,
                "Scenario": r.get("domain", "N/A"),
                "Num_Injected_Tools": r.get("num_injected_tools", 0),
                "Benign_Tools": r.get("num_benign_tools", 0),
                "Total_Queries": test_stats.get("total_queries", 0),
                "ASR": f"{test_stats.get('asr', 0.0):.4f}",
                "TDR": f"{test_stats.get('tdr', 0.0):.4f}",
                "Mean_Domination": f"{test_stats.get('mean_domination', 0.0):.4f}",
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


def generate_attacker_tools_for_domain(
    task_name: Optional[str],
    exp_cfg,
    poisonrag_cfg,
    benign_data_dir: Path,
    benign_tools: List[Tool],
    attack_embedding_model,
    attack_embedding_model_name: str,
    llm_generator: Any,
) -> Tuple[List[Tool], List[str], List[str], str, Dict[str, Any]]:
    """Generate attacker tools for a task.
    
    Returns tools, train/test queries, task_str, and attack_details.
    """
    task_str = task_name if task_name else "all"
    logger.info(f"Generating attacker tools for task: {task_str}")

    # Load queries from tasks folder
    tasks_path = benign_data_dir / "tasks"
    task_list = [task_name] if task_name else None
    train_queries, test_queries = load_queries_from_tasks(tasks_path, task_names=task_list)
    train_queries, test_queries = limit_queries(
        train_queries,
        test_queries,
        exp_cfg.max_train_queries,
        exp_cfg.max_test_queries,
    )
    logger.info(
        f"Task {task_str}: {len(train_queries)} train, {len(test_queries)} test queries"
    )

    # Run PoisonRAG attack (config from YAML; runner sets max_train_queries from experiment)
    pr_cfg = PoisonRAGConfig(
        **poisonrag_cfg.model_dump(),
        max_train_queries=exp_cfg.max_train_queries,
    )
    attack = PoisonRAGBlackBoxAttack(
        train_queries,
        benign_tools,
        attack_embedding_model,
        pr_cfg,
        llm_generator=llm_generator,
    )
    attacker_tools, attack_details = attack.attack()
    
    return attacker_tools, train_queries, test_queries, task_str, attack_details


def prepare_vector_store_for_embedding(
    task_str: str,
    attacker_tools: List[Tool],
    output_dir: Path,
    victim_embedding_model,
    victim_embedding_model_name: str,
    attack_embedding_model_name: str,
    benign_tools: List[Tool],
    attack_details: Dict[str, Any],
) -> Tuple[Path, List[Tool], Set[str]]:
    """Prepare vector store and merged tools for an embedding combination.
    
    This is done once per embedding combination and reused across all models.
    Returns (vectorstore_path, merged_tools, attacker_tool_names).
    """
    # Output directory structure matches ToolFlood runner:
    # output_dir/<task>/<attack_emb_..._victim_emb_...>/
    task_dir = output_dir / task_str if task_str != "all" else output_dir
    embedding_subdir = (
        f"attack_emb_{attack_embedding_model_name}_"
        f"victim_emb_{victim_embedding_model_name}"
    )
    run_dir = task_dir / embedding_subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save attack artifacts (match ToolFlood naming)
    with (run_dir / "attack_tools.json").open("w", encoding="utf-8") as f:
        json.dump({t.name: t.description for t in attacker_tools}, f, indent=2)
    with (run_dir / "attack_details.json").open("w", encoding="utf-8") as f:
        json.dump(attack_details, f, indent=2)

    # Merge tools and build (or reuse) vector store for this embedding combo
    merged_tools, attacker_names = merge_tools(
        benign_tools,
        attacker_tools,
        run_dir / "merged_tools.json",
        attacker_label="poisoned",
    )
    vectorstore_path = run_dir / "vectorstore"
    init_vector_store(
        merged_tools,
        victim_embedding_model,
        vectorstore_path=vectorstore_path,
        force_rebuild=False,  # Reuse existing if it exists
    )
    
    return vectorstore_path, merged_tools, attacker_names


async def run_experiment_for_model(
    model_name: str,
    task_str: str,
    test_queries: List[str],
    cfg: Dict,
    exp_cfg,
    agent_cfg,
    victim_embedding_model,
    victim_embedding_model_name: str,
    attack_embedding_model_name: str,
    vectorstore_path: Path,
    merged_tools: List[Tool],
    attacker_tool_names: Set[str],
) -> Dict[str, Any]:
    """Run experiment for a specific model.
    
    Uses pre-generated attacker tools and pre-built vector store.
    """
    logger.info(
        f"Evaluating model: {model_name} with embedding: "
        f"{victim_embedding_model_name} for task: {task_str}"
    )

    # Load existing vector store (shared across models with same embedding)
    merged_vectorstore = load_vector_store(vectorstore_path, victim_embedding_model)
    victim_llm = init_llm(cfg, model_name=model_name)
    
    # Agent retrieval depth is configured under `agent.top_k` in config.
    # We keep `experiment.max_train_evaluation_queries` as a legacy fallback.
    top_k = agent_cfg.top_k
    agent = VictimAgent(
        tools=merged_tools,
        vectorstore=merged_vectorstore,
        embedding_model=victim_embedding_model,
        llm=victim_llm,
        top_k=top_k,
        verbose=agent_cfg.verbose,
    )

    eval_results = await evaluate_queries(agent, test_queries, attacker_tool_names)

    asr = calculate_asr(
        eval_results["attacker_selections"],
        eval_results["benign_selections"],
        eval_results["no_selection"],
    )
    tdr = calculate_tdr(eval_results["top_k_tools"], attacker_tool_names, k=top_k)
    mean_dom = calculate_mean_domination(
        eval_results["top_k_tools"], attacker_tool_names, k=top_k
    )

    result: Dict[str, Any] = {
        "domain": task_str,
        "model": model_name,
        "attack_embedding_model": attack_embedding_model_name,
        "victim_embedding_model": victim_embedding_model_name,
        "top_k": top_k,
        "num_injected_tools": len(attacker_tool_names),
        "num_benign_tools": len(merged_tools) - len(attacker_tool_names),
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

    return result


async def _run_single_experiment(cfg_path: Path, models_path: Path) -> int:
    """Run PoisonRAG experiments from config and models files, organized by task."""
    cfg = load_config(cfg_path)
    models_cfg = load_models(models_path)
    full_cfg = {**cfg, **models_cfg}
    exp_cfg = load_experiment_config(cfg_path)
    agent_cfg = load_agent_config(cfg_path)
    poisonrag_cfg = load_poisonrag_config(cfg_path)

    base_path = get_base_path(cfg_path)
    benign_data_dir = resolve_path(base_path, exp_cfg.benign_data_directory)
    output_dir = resolve_path(base_path, exp_cfg.output_directory)
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
        all_results,
        ["domain", "model", "attack_embedding_model", "victim_embedding_model"],
    )
    if completed:
        logger.info(f"Found {len(completed)} completed combinations; will skip them.")

    # Load benign tools
    benign_tools = load_tools(benign_data_dir / "tools.json")
    logger.info(f"Loaded {len(benign_tools)} benign tools")

    # Initialize LLM generator for attack (from attack config)
    generator_model_name = poisonrag_cfg.generator_model
    if not generator_model_name:
        raise ValueError(
            "generator_model is required in poisonrag_attack config. "
            "Provide an LLM model name for generating tool descriptions."
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

    # Grids come from experiment config (same as ToolFlood).
    attack_embedding_model_names = exp_cfg.attack_embedding_models
    victim_embedding_model_names = exp_cfg.victim_embedding_models
    victim_model_names = exp_cfg.victim_models

    # Initialize embedding models (cache)
    attack_embedding_models: Dict[str, Any] = {}
    victim_embedding_models: Dict[str, Any] = {}
    for name in attack_embedding_model_names:
        attack_embedding_models[name] = init_embedding_model(
            full_cfg, model_name=name
        )
    for name in victim_embedding_model_names:
        victim_embedding_models[name] = init_embedding_model(
            full_cfg, model_name=name
        )

    total_combinations = (
        len(tasks)
        * len(attack_embedding_model_names)
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
        
        for attack_emb_name in attack_embedding_model_names:
            attack_embedding_model = attack_embedding_models[attack_emb_name]

            # Check if there are any remaining combinations for this task/attack_emb pair
            has_remaining_combinations = False
            for victim_emb_name in victim_embedding_model_names:
                for model_name in victim_model_names:
                    combination_key = (
                        task_str, model_name,
                        attack_emb_name, victim_emb_name
                    )
                    if combination_key not in completed:
                        has_remaining_combinations = True
                        break
                if has_remaining_combinations:
                    break
            
            # Skip generating attacker tools if all combinations are already done
            if not has_remaining_combinations:
                logger.info(
                    f"Skipping task '{task_str}' with attack_emb '{attack_emb_name}': "
                    "all combinations already completed"
                )
                continue

            # Generate attacker tools once per task and attack embedding model
            attacker_tools, train_queries, test_queries, task_str, attack_details = (
                generate_attacker_tools_for_domain(
                    task_name,
                    exp_cfg,
                    poisonrag_cfg,
                    benign_data_dir,
                    benign_tools,
                    attack_embedding_model,
                    attack_emb_name,
                    llm_generator,
                )
            )

            # Test all victim models and victim embedding model combinations
            # Build vector store once per victim embedding model (shared across all victim LLM models)
            for victim_emb_name in victim_embedding_model_names:
                victim_embedding_model = victim_embedding_models[victim_emb_name]
                
                # Prepare vector store once per embedding combination
                # This will be reused across all victim LLM models with the same embedding
                vectorstore_path, merged_tools, attacker_tool_names = (
                    prepare_vector_store_for_embedding(
                        task_str,
                        attacker_tools,
                        output_dir,
                        victim_embedding_model,
                        victim_emb_name,
                        attack_emb_name,
                        benign_tools,
                        attack_details,
                    )
                )
                
                # Test all victim LLM models with this embedding model
                for model_name in victim_model_names:
                    # Check if this combination is already completed
                    combination_key = (
                        task_str, model_name,
                        attack_emb_name, victim_emb_name
                    )
                    if combination_key in completed:
                        logger.info(
                            f"Skipping already completed: {task_str} / "
                            f"{model_name} / attack_emb: "
                            f"{attack_emb_name} / victim_emb: "
                            f"{victim_emb_name}"
                        )
                        continue

                    logger.info(
                        f"Running experiment: {task_str} / {model_name} / "
                        f"attack_emb: {attack_emb_name} / "
                        f"victim_emb: {victim_emb_name} "
                        f"({completed_count + 1}/{total_combinations})"
                    )

                    try:
                        result = await run_experiment_for_model(
                            model_name,
                            task_str,
                            test_queries,
                            full_cfg,
                            exp_cfg,
                            agent_cfg,
                            victim_embedding_model,
                            victim_emb_name,
                            attack_emb_name,
                            vectorstore_path,
                            merged_tools,
                            attacker_tool_names,
                        )
                        all_results.append(result)
                        completed.add(combination_key)
                        completed_count += 1

                        # Save results incrementally after each experiment
                        save_results(
                            all_results,
                            output_dir,
                            exp_cfg,
                            results_json_path,
                            results_table_path,
                        )
                        logger.success(
                            f"Completed: {task_str} / {model_name} / "
                            f"attack_emb: {attack_emb_name} / "
                            f"victim_emb: {victim_emb_name} "
                            f"({completed_count}/{total_combinations})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error running experiment for "
                            f"{task_str} / {model_name} / "
                            f"attack_emb: {attack_emb_name} / "
                            f"victim_emb: {victim_emb_name}: {e}"
                        )
                        logger.error("Results saved up to this point.")
                        raise

    # Final save and summary
    if all_results:
        save_results(
            all_results,
            output_dir,
            exp_cfg,
            results_json_path,
            results_table_path,
        )

        # Print summary
        results_table = pd.read_csv(results_table_path)
        logger.success("\n" + "="*100)
        logger.success("RESULTS TABLE")
        logger.success("="*100)
        print("\n" + results_table.to_string(index=False))
        logger.success("="*100)
        logger.success(f"\nResults table saved to: {results_table_path}")
    else:
        logger.warning("No results to display.")

    logger.success("All PoisonRAG experiments completed.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run black-box PoisonRAG experiment.",
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