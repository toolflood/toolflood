#!/usr/bin/env python3
"""Utility functions for config, loading, and initialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import init_embeddings
from pydantic import BaseModel, Field
import pandas as pd


# Pydantic models for type safety and validation
class Tool(BaseModel):
    """Represents a tool with its metadata."""
    tool_id: str
    name: str
    description: str


class ToolDecision(BaseModel):
    """Represents the LLM's decision on which tool to choose."""
    chosen_tool_id: Optional[str] = None
    reason: str = Field(default="chosen by LLM")


class GeneratedTool(BaseModel):
    """Represents a generated tool with name and description."""
    name: str
    description: str


class GeneratedTools(BaseModel):
    """Represents a list of generated tools."""
    tools: List[GeneratedTool]


class ToolFloodConfig(BaseModel):
    """Flat configuration for ToolFlood (loaded from config.toolflood)."""
    total_tool_budget: Optional[int] = Field(
        default=None,
        description=(
            "Budget for total number of tools to generate. "
            "If set, limits the number of tools generated"
        )
    )
    attacker_tools_output_path: str = Field(
        description="Output JSON path for generated tools"
    )
    max_embedding_distance: float = Field(
        default=0.4,
        description="Maximum allowed embedding distance between query and generated tool."
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description=(
            "Name of embedding model from embeddings dictionary to use. "
            "If None, uses first embedding model in embeddings dict"
        )
    )
    llm_optimizer_model: Optional[str] = Field(
        default=None,
        description=(
            "Name of model from models dictionary to use as optimizer. "
            "If None, uses first model in models dict"
        )
    )
    num_tools_per_query: int = Field(
        description="Number of generated tools per query (top-k)"
    )
    query_sample_size: int = Field(
        default=10,
        description="Number of queries to sample in each Phase 1 iteration"
    )
    num_tools_per_sample: Optional[int] = Field(
        default=None,
        description=(
            "Number of tools to generate per sample before filtering. "
            "If None, defaults to num_tools_per_query"
        )
    )
    max_generation_iterations: int = Field(
        default=20,
        description="Maximum number of iterations for Phase 1"
    )
    max_concurrent_tasks: int = Field(
        default=5,
        description="Maximum number of concurrent tasks in Phase 1 (parallelization)"
    )


class ExperimentConfig(BaseModel):
    """Configuration for experiment script."""
    benign_data_directory: str = Field(
        description=(
            "Directory containing benign data "
            "(tasks/ folder or tasks.json, tools.json, vectorstore/)"
        )
    )
    output_directory: str = Field(
        default="./experiment_output",
        description="Directory to store experiment outputs"
    )
    max_train_queries: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of train queries to use for tool generation"
        )
    )
    max_test_queries: Optional[int] = Field(
        default=None,
        description="Maximum number of test queries to evaluate"
    )
    max_train_evaluation_queries: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of train queries to use for evaluation. "
            "If None, evaluates on all train queries. "
            "If set, randomly samples this many queries from train set."
        )
    )
    victim_models: List[str] = Field(
        default=["gpt-4o-mini"],
        description="List of model names to evaluate"
    )
    attack_embedding_models: List[str] = Field(
        default=["text-embedding-3-small"],
        description=(
            "List of embedding model names from embeddings dictionary to use "
            "for generating tools. Each model will be used to "
            "generate tools separately"
        )
    )
    victim_embedding_models: List[str] = Field(
        default=["text-embedding-3-small"],
        description=(
            "List of embedding model names from embeddings dictionary to use "
            "for retrieval. Each model will be tested with each "
            "evaluation model"
        )
    )
    task_names: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of task names from tasks folder or "
            "tasks.json to filter queries by. "
            "If None, all tasks are used. "
            "Examples: ['Space images', "
            "'Website performance + SEO + keywords']"
        )
    )
    hard_reset: bool = Field(
        default=False,
        description=(
            "If True, clears existing results and starts fresh. "
            "If False, loads existing results and continues from "
            "where it left off."
        )
    )


class AgentConfig(BaseModel):
    """Configuration for agent script."""
    query: Optional[str] = Field(default=None, description="Query to test")
    verbose: bool = Field(default=True, description="Enable verbose logging")
    top_k: int = Field(
        default=10, description="Number of top tools to retrieve"
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description=(
            "Name of embedding model from embeddings dictionary to use. "
            "If None, uses first embedding model in embeddings dict."
        )
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file (no models/embeddings)."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models(models_path: Path) -> Dict[str, Any]:
    """Load models and embeddings from YAML file.

    Returns a dict with keys 'models' and 'embeddings', to be merged
    with config for init_llm / init_embedding_model.
    """
    with models_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "models": data.get("models", {}),
        "embeddings": data.get("embeddings", {}),
    }


def load_toolflood_config(config_path: Path) -> ToolFloodConfig:
    """Load ToolFlood configuration from YAML file (config.toolflood)."""
    cfg = load_config(config_path)
    toolflood_dict = cfg.get("toolflood", {})
    return ToolFloodConfig(**toolflood_dict)


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    cfg = load_config(config_path)
    exp_cfg = cfg.get("experiment", {})
    return ExperimentConfig(**exp_cfg)


def load_agent_config(config_path: Path) -> AgentConfig:
    """Load agent configuration from YAML file."""
    cfg = load_config(config_path)
    agent_cfg = cfg.get("agent", {})
    return AgentConfig(**agent_cfg)


def get_base_path(config_path: Path) -> Path:
    """Get base path from config file location (project root)."""
    parent = config_path.parent
    if parent.name in ("config", ".config"):
        return parent.parent
    if parent.name == "configs":
        return parent.parent.parent
    return parent


def resolve_path(base: Path, maybe_rel: str) -> Path:
    """Resolve a path relative to base if not absolute."""
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base / p).resolve()


def get_data_dir(config_path: Path, cfg: Dict[str, Any]) -> Path:
    """Get data directory path from config."""
    base = get_base_path(config_path)
    data_dir = cfg.get("paths", {}).get("data_dir", "./data")
    return resolve_path(base, data_dir)


def init_embedding_model(
    cfg: Dict[str, Any], model_name: Optional[str] = None
) -> Any:
    """Initialize embedding model from config.

    Args:
        cfg: Configuration dictionary
        model_name: Name of embedding model to use. If None, uses first model.
    """
    embeddings_dict = cfg.get("embeddings", {})
    if not embeddings_dict:
        raise ValueError("No 'embeddings' configuration found")

    # If model_name not provided, use first model
    if model_name is None:
        model_name = list(embeddings_dict.keys())[0]

    if model_name not in embeddings_dict:
        available = list(embeddings_dict.keys())
        raise ValueError(
            f"Embedding model '{model_name}' not found. Available: {available}"
        )

    emb_cfg = embeddings_dict[model_name]
    return init_embeddings(**emb_cfg)


def load_tools(tools_path: Path) -> List[Tool]:
    """Load tools from JSON file and return as list of Tool models."""
    tools_dict = json.loads(tools_path.read_text(encoding="utf-8"))
    return [
        Tool(tool_id=name, name=name, description=description)
        for name, description in tools_dict.items()
    ]


def load_tools_categorized(
    categorized_path: Path
) -> Dict[str, Dict[str, str]]:
    """
    Load categorized tools from JSON file.

    Args:
        categorized_path: Path to tools_categorized.json

    Returns:
        Dictionary mapping category names to their tools dictionary
    """
    return json.loads(categorized_path.read_text(encoding="utf-8"))


def get_tools_from_categories(
    categorized_path: Path,
    category_names: List[str]
) -> List[str]:
    """
    Get tool names from specified categories.

    Args:
        categorized_path: Path to tools_categorized.json
        category_names: List of category names
            (e.g., ['research_and_academic'])

    Returns:
        List of tool names belonging to the specified categories
    """
    categorized = load_tools_categorized(categorized_path)
    tool_names = []

    for category_name in category_names:
        if category_name not in categorized:
            raise ValueError(
                f"Category '{category_name}' not found in categorized tools. "
                f"Available categories: {list(categorized.keys())}"
            )
        category_data = categorized[category_name]
        tools = category_data.get("tools", {})
        tool_names.extend(tools.keys())

    return tool_names


def load_queries_from_tasks(
    tasks_path: Path,
    task_names: Optional[List[str]] = None
) -> tuple[List[str], List[str]]:
    """
    Load train and test queries from tasks folder or tasks.json file.

    Args:
        tasks_path: Path to tasks folder or tasks.json file
        task_names: Optional list of task names to filter by.
            If None, all tasks are used.

    Returns:
        Tuple of (train_queries, test_queries) lists
    """
    tasks = []

    # Check if tasks_path is a directory (tasks folder) or a file (tasks.json)
    if tasks_path.is_dir():
        # Load all JSON files from the tasks folder
        task_files = sorted(tasks_path.glob("*.json"))
        for task_file in task_files:
            task_data = json.loads(task_file.read_text(encoding="utf-8"))
            tasks.append(task_data)
    elif tasks_path.is_file():
        # Backward compatibility: load from tasks.json file
        tasks_data = json.loads(tasks_path.read_text(encoding="utf-8"))
        tasks = tasks_data.get("tasks", [])
    else:
        raise FileNotFoundError(f"Tasks path not found: {tasks_path}")

    train_queries = []
    test_queries = []

    for task in tasks:
        task_name = task.get("task_name")
        if task_names is None or task_name in task_names:
            train_queries.extend(task.get("train_queries", []))
            test_queries.extend(task.get("test_queries", []))

    return train_queries, test_queries


def load_queries(
    queries_path: Path,
    domain: Optional[List[str]] = None,
    categorized_tools_path: Optional[Path] = None
) -> List[str]:
    """
    Load queries from CSV file.

    Args:
        queries_path: Path to CSV file with 'Query' and 'Tool' columns
        domain: Optional list of domain names (category names or tool names)
            to filter by. If categorized_tools_path is provided, domain is
            treated as category names. Otherwise, domain is treated as tool
            names. If None, all queries are returned.
        categorized_tools_path: Optional path to tools_categorized.json.
            If provided, domain will be interpreted as category names.

    Returns:
        List of query strings
    """
    queries_df = pd.read_csv(queries_path)

    # Filter by domain if specified
    if domain is not None:
        if (
            categorized_tools_path is not None
            and categorized_tools_path.exists()
        ):
            # Convert category names to tool names
            tool_names = get_tools_from_categories(
                categorized_tools_path, domain
            )
            queries_df = queries_df[queries_df["Tool"].isin(tool_names)]
        else:
            # Treat domain as tool names directly
            queries_df = queries_df[queries_df["Tool"].isin(domain)]

    # Remove duplicate queries
    queries_df = queries_df.drop_duplicates(subset=["Query"])

    return queries_df["Query"].tolist()


def load_vector_store(vectorstore_path: Path, embedding_model: Any) -> FAISS:
    """Load FAISS vector store from disk."""
    if not vectorstore_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {vectorstore_path}. "
            "Please run scripts/build_vectorstore.py first."
        )
    return FAISS.load_local(
        str(vectorstore_path),
        embedding_model,
        allow_dangerous_deserialization=True
    )


def init_llm(cfg: Dict[str, Any], model_name: Optional[str] = None) -> Any:
    """Initialize LLM from config.

    Args:
        cfg: Configuration dictionary
        model_name: Name of model to use. If None, uses first model.
    """
    llms_dict = cfg.get("models", {})
    if not llms_dict:
        raise ValueError("No 'models' configuration found")

    # If model_name not provided, use first model
    if model_name is None:
        model_name = list(llms_dict.keys())[0]

    if model_name not in llms_dict:
        available = list(llms_dict.keys())
        raise ValueError(
            f"Model '{model_name}' not found. Available: {available}"
        )

    llm_cfg = llms_dict[model_name].copy()

    return init_chat_model(**llm_cfg)


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine similarity)."""
    u_norm = u / (np.linalg.norm(u) + 1e-10)
    v_norm = v / (np.linalg.norm(v) + 1e-10)
    return 1.0 - float(np.dot(u_norm, v_norm))
