#!/usr/bin/env python3
"""
Analyze ASR (Alternative Selection Rate) per task for varying B levels.

This script:
1. Loads experiment results from experiment_output/
2. For each task, ranks generated tools by phase2_query_coverage
3. For each B level, selects top B tools
4. Creates merged_tools.json and vector store for each B level
5. Creates agent and re-evaluates on train and test queries
6. Calculates ASR, TDR, Mean Domination per task per B level
7. Creates CSV and plot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Add repo root to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document

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
    load_models,
    load_queries_from_tasks,
    load_tools,
    load_vector_store,
    resolve_path,
)
from src.scripts.build_vectorstore import init_vector_store


def load_attack_tools(attack_tools_path: Path) -> Dict[str, Dict]:
    """Load generated tools JSON file."""
    with open(attack_tools_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_top_b_tools(attack_tools: Dict[str, Dict], b: int) -> List[str]:
    """
    Get top B tools ranked by phase2_query_coverage.
    
    Args:
        attack_tools: Dictionary of tool_name -> {description, phase2_query_coverage} (generated tools)
        b: Number of top tools to select
        
    Returns:
        List of top B tool names
    """
    # Sort tools by phase2_query_coverage (descending)
    sorted_tools = sorted(
        attack_tools.items(),
        key=lambda x: x[1].get("phase2_query_coverage", 0),
        reverse=True
    )
    
    # Get top B tool names
    top_b = [tool_name for tool_name, _ in sorted_tools[:b]]
    
    return top_b


def create_merged_tools_list(
    benign_tools: List[Tool],
    attack_tools: Dict[str, Dict],
    top_b_tool_names: List[str],
    output_path: Path
) -> tuple[List[Tool], Set[str]]:
    """
    Create merged tools list with benign tools + top B generated tools.
    
    Args:
        benign_tools: List of benign tools
        attack_tools: Dictionary of generated tool_name -> {description, ...}
        top_b_tool_names: List of top B generated tool names to include
        output_path: Path to save merged_tools.json
        
    Returns:
        Tuple of (merged_tools_list, generated_tool_names_set)
    """
    merged_dict = {tool.name: tool.description for tool in benign_tools}
    attacker_names = set()
    
    for tool_name in top_b_tool_names:
        if tool_name in attack_tools:
            original_name = tool_name
            tool_name_final = original_name
            
            # Handle name conflicts
            if tool_name_final in merged_dict:
                suffix = 1
                while f"{original_name}_{suffix}" in merged_dict:
                    suffix += 1
                tool_name_final = f"{original_name}_{suffix}"
            
            merged_dict[tool_name_final] = attack_tools[tool_name]["description"]
            attacker_names.add(tool_name_final)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged tools
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_dict, f, indent=2)
    
    # Convert to Tool objects
    merged_tools = [
        Tool(tool_id=name, name=name, description=description)
        for name, description in merged_dict.items()
    ]
    
    logger.info(
        f"Created merged_tools.json with {len(benign_tools)} benign + "
        f"{len(top_b_tool_names)} generated = {len(merged_tools)} total tools"
    )
    
    return merged_tools, attacker_names


async def evaluate_queries(
    agent: VictimAgent,
    queries: List[str],
    attacker_tool_names: Set[str],
) -> Dict:
    """Evaluate queries on the agent and track selections."""
    logger.info(f"Evaluating {len(queries)} queries...")

    results = {
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
            top_k_tool_names = [tool.name for tool in top_k_tools]
            results["top_k_tools"].append(top_k_tool_names)

            selected_tool = await agent.llm_choose_tool(query, top_k_tools)

            if selected_tool is None:
                results["no_selection"] += 1
                results["query_results"].append({
                    "query": query,
                    "selected_tool": None,
                    "is_attacker": False,
                })
            else:
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
        except Exception as e:
            logger.error(f"Error processing query {i+1}: {e}")
            results["no_selection"] += 1
            results["top_k_tools"].append([])
            results["query_results"].append({
                "query": query,
                "selected_tool": None,
                "is_attacker": False,
                "error": str(e),
            })

    return results


def load_results_full(results_path: Path) -> List[Dict]:
    """Load results_full.json file."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ASR per task for varying B levels"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("config/models.yaml"),
        help="Path to models YAML file",
    )
    args = parser.parse_args()

    # Load config and models
    cfg_path = args.config.resolve()
    models_path = args.models.resolve()
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        return
    if not models_path.exists():
        logger.error(f"Models file not found: {models_path}")
        return

    cfg = load_config(cfg_path)
    models_cfg = load_models(models_path)
    full_cfg = {**cfg, **models_cfg}
    b_level_cfg = cfg.get("b_level_analysis", {})
    exp_cfg = cfg.get("experiment", {})
    agent_cfg_dict = cfg.get("agent", {})
    
    # Get base path for resolving relative paths
    base_path = get_base_path(cfg_path)
    
    # Get parameters from config with defaults
    experiment_output = resolve_path(
        base_path,
        b_level_cfg.get("experiment_output", "./experiment_output")
    )
    output_dir = resolve_path(
        base_path,
        b_level_cfg.get("output_directory", "./experiment_output_b_levels")
    )
    benign_tools_path = resolve_path(
        base_path,
        b_level_cfg.get("benign_tools_path", "./data/ToolBench/tools.json")
    )
    benign_data_dir = resolve_path(
        base_path,
        exp_cfg.get("benign_data_directory", "./data/ToolBench")
    )
    b_levels = sorted(b_level_cfg.get("b_levels", [5, 10, 15, 20]))
    
    # Get evaluation model and embedding from config (use first ones)
    victim_models = exp_cfg.get("victim_models", ["gpt-4o-mini"])
    victim_embedding_models = exp_cfg.get("victim_embedding_models", ["text-embedding-3-small"])
    victim_model = victim_models[0]
    victim_embedding_model_name = victim_embedding_models[0]
    
    # Get top_k from agent config
    top_k = agent_cfg_dict.get("top_k", 5)
    
    logger.info(f"Experiment output: {experiment_output}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"B levels: {b_levels}")
    logger.info(f"Evaluation model: {victim_model}")
    logger.info(f"Evaluation embedding: {victim_embedding_model_name}")
    logger.info(f"Top K: {top_k}")
    
    # Load benign tools
    logger.info("Loading benign tools...")
    benign_tools = load_tools(benign_tools_path)
    logger.info(f"Loaded {len(benign_tools)} benign tools")
    
    # Load results_full.json to get task domains
    results_path = experiment_output / "results_full.json"
    if not results_path.exists():
        logger.error(f"results_full.json not found at {results_path}")
        return
    
    logger.info("Loading results_full.json...")
    all_results = load_results_full(results_path)
    logger.info(f"Loaded {len(all_results)} task results")
    
    # Organize results by task domain
    results_by_domain = {}
    for result in all_results:
        domain = result.get("domain")
        if domain:
            if domain not in results_by_domain:
                results_by_domain[domain] = []
            results_by_domain[domain].append(result)
    
    # Initialize embedding model
    victim_embedding_model = init_embedding_model(
        full_cfg, victim_embedding_model_name
    )

    # Initialize LLM
    llm = init_llm(full_cfg, victim_model)
    
    # Initialize agent config
    agent_cfg = load_agent_config(cfg_path)
    
    # Prepare data for CSV
    csv_data = []
    
    # Process each task
    for domain in tqdm(sorted(results_by_domain.keys()), desc="Processing tasks"):
        task_results = results_by_domain[domain]
        
        # Find the result with generated tools
        result_with_attack = None
        attack_tools_path = None
        
        # Try to find generated tools JSON for this domain
        for result in task_results:
            # Construct path to attack_tools.json (file name unchanged)
            domain_dir = experiment_output / domain
            attack_emb = result.get("attack_embedding_model", "text-embedding-3-small")
            victim_emb = result.get("victim_embedding_model", "text-embedding-3-small")
            attack_dir = domain_dir / f"attack_emb_{attack_emb}_victim_emb_{victim_emb}"
            potential_attack_tools_path = attack_dir / "attack_tools.json"
            
            if potential_attack_tools_path.exists():
                attack_tools_path = potential_attack_tools_path
                result_with_attack = result
                break
        
        if attack_tools_path is None or not attack_tools_path.exists():
            logger.warning(f"No generated tools JSON found for domain: {domain}")
            continue
        
        logger.info(f"\nProcessing domain: {domain}")
        
        # Load generated tools
        attack_tools_dict = load_attack_tools(attack_tools_path)
        logger.info(f"Loaded {len(attack_tools_dict)} generated tools")
        
        # Load test queries for this task
        task_names = [domain] if domain else None
        _, test_queries = load_queries_from_tasks(
            benign_data_dir / "tasks",
            task_names=task_names
        )
        
        # Apply limits if specified
        max_test = exp_cfg.get("max_test_queries")
        
        if max_test and len(test_queries) > max_test:
            import random
            test_queries = random.sample(test_queries, max_test)
        
        logger.info(f"Test queries: {len(test_queries)}")
        
        # Create base vector store with benign tools (once per domain)
        base_vectorstore_path = output_dir / domain / "base_vectorstore"
        logger.info("Creating base vector store with benign tools...")
        init_vector_store(
            benign_tools,
            victim_embedding_model,
            base_vectorstore_path,
            force_rebuild=False
        )
        base_vectorstore = load_vector_store(base_vectorstore_path, victim_embedding_model)
        
        # Track which generated tools have been added to vector store
        previous_attacker_names = set()
        
        # Process each B level
        for b_level in b_levels:
            logger.info(f"\nProcessing B={b_level} for domain: {domain}")
            
            # Get top B tools
            top_b_tool_names = get_top_b_tools(attack_tools_dict, b_level)
            logger.info(f"Selected top {b_level} tools")
            
            # Create merged tools for this B level
            merged_dir = output_dir / domain / f"B_{b_level}"
            merged_path = merged_dir / "merged_tools.json"
            vectorstore_path = merged_dir / "vectorstore"
            
            merged_tools, attacker_tool_names = create_merged_tools_list(
                benign_tools,
                attack_tools_dict,
                top_b_tool_names,
                merged_path
            )
            
            # Build vector store incrementally
            # Find new generated tools by comparing with previous iteration
            new_attacker_tools = attacker_tool_names - previous_attacker_names
            
            if new_attacker_tools:
                logger.info(f"Adding {len(new_attacker_tools)} new generated tools to vector store...")
                # Create documents for new tools
                new_tools_docs = []
                for tool_name in new_attacker_tools:
                    # Find the corresponding Tool object from merged_tools
                    for merged_tool in merged_tools:
                        if merged_tool.name == tool_name:
                            new_tools_docs.append(
                                Document(
                                    page_content=f"{merged_tool.name}: {merged_tool.description}",
                                    metadata={
                                        "tool_name": merged_tool.name,
                                        "description": merged_tool.description
                                    }
                                )
                            )
                            break
                
                # Add new tools to vector store
                if new_tools_docs:
                    base_vectorstore.add_documents(new_tools_docs)
            else:
                logger.info("No new tools to add")
            
            # Update previous generated tool names for next iteration
            previous_attacker_names = attacker_tool_names.copy()
            
            # Save vector store for this B level
            vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
            base_vectorstore.save_local(str(vectorstore_path))
            
            # Create agent (reuse the same vector store instance)
            vectorstore = base_vectorstore
            
            agent = VictimAgent(
                tools=merged_tools,
                vectorstore=vectorstore,
                embedding_model=victim_embedding_model,
                llm=llm,
                top_k=top_k,
                verbose=False,
            )
            
            # Evaluate on test queries
            logger.info("Evaluating test queries...")
            test_results = asyncio.run(
                evaluate_queries(agent, test_queries, attacker_tool_names)
            )
            
            test_asr = calculate_asr(
                test_results["attacker_selections"],
                test_results["benign_selections"],
                test_results["no_selection"],
            )
            test_tdr = calculate_tdr(
                test_results["top_k_tools"],
                attacker_tool_names,
                top_k
            )
            test_mean_dom = calculate_mean_domination(
                test_results["top_k_tools"],
                attacker_tool_names,
                top_k
            )
            
            logger.info(
                f"Domain: {domain}, B: {b_level} - "
                f"ASR: {test_asr:.4f}, TDR: {test_tdr:.4f}, "
                f"Mean Dom: {test_mean_dom:.4f}"
            )
            
            # Add to CSV data
            csv_data.append({
                "Domain": domain,
                "B": b_level,
                "ASR": test_asr,
                "TDR": test_tdr,
                "Mean_Domination": test_mean_dom,
                "Num_Attack_Tools": len(top_b_tool_names),
                "Num_Benign_Tools": len(benign_tools),
                "Total_Tools": len(merged_tools),
            })
            
            # Delete vector store for this B level after analysis
            if vectorstore_path.exists():
                logger.info(f"Deleting vector store for B={b_level}...")
                shutil.rmtree(vectorstore_path)
                logger.info(f"Deleted vector store at {vectorstore_path}")
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "asr_by_b_level.csv"
    df.to_csv(csv_path, index=False)
    logger.success(f"\nSaved CSV to {csv_path}")
    
    # Create plot for ASR
    if len(csv_data) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Pivot data for plotting
        pivot_df = df.pivot(index="Domain", columns="B", values="ASR")
        
        # Plot
        pivot_df.plot(kind="bar", ax=ax, width=0.8)
        ax.set_xlabel("Task Domain", fontsize=12)
        ax.set_ylabel("ASR (Alternative Selection Rate)", fontsize=12)
        ax.set_title("ASR per Task for Varying B Levels (Test)", fontsize=14, fontweight="bold")
        ax.legend(title="B Level", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plot_path = output_dir / "asr_by_b_level_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.success(f"Saved plot to {plot_path}")
        
        # Also create a line plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        for domain in pivot_df.index:
            ax2.plot(pivot_df.columns, pivot_df.loc[domain], marker="o", label=domain, linewidth=2)
        
        ax2.set_xlabel("B Level", fontsize=12)
        ax2.set_ylabel("ASR (Alternative Selection Rate)", fontsize=12)
        ax2.set_title("ASR vs B Level per Task (Test)", fontsize=14, fontweight="bold")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(alpha=0.3)
        ax2.set_xticks(b_levels)
        plt.tight_layout()
        
        plot_path2 = output_dir / "asr_vs_b_level_lines.png"
        plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
        logger.success(f"Saved line plot to {plot_path2}")
    
    logger.success("\nAnalysis complete!")


if __name__ == "__main__":
    main()
