#!/usr/bin/env python3
"""
ToolFlood: Candidate capture on tool-using agents.

Phase 1: Generate tools from sampled queries (parallelized).
Phase 2: Greedily select tools that maximize query coverage.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from tqdm.asyncio import tqdm

from src.utils import GeneratedTools, Tool, cosine_distance


@dataclass
class ToolCandidate:
    """Tool with metadata for selection."""
    name: str
    description: str
    min_distance_to_queries: float
    embedding_vector: np.ndarray


@dataclass
class AttackConfig:
    """Configuration for ToolFlood."""
    num_tools_per_query: int
    query_sample_size: int = 10
    num_tools_per_sample: Optional[int] = None
    max_generation_iterations: int = 20
    max_embedding_distance: float = 0.4
    total_tool_budget: Optional[int] = None
    max_concurrent_tasks: int = 5

    def __post_init__(self):
        if self.num_tools_per_sample is None:
            self.num_tools_per_sample = self.num_tools_per_query


class ToolFloodAttack:
    """ToolFlood implementation with optimized performance."""

    def __init__(
        self,
        target_queries: List[str],
        embedding_model,
        llm_optimizer,
        attack_config: Optional[AttackConfig] = None,
        **config_kwargs,
    ):
        """
        Initialize pipeline.
        
        Args:
            target_queries: Target queries to cover
            embedding_model: Embedding function
            llm_optimizer: LLM for tool generation
            attack_config: ToolFlood configuration (or pass config_kwargs)
        """
        self.target_queries = target_queries
        self.embedding_model = embedding_model
        self.llm_optimizer = llm_optimizer
        self.attack_config = attack_config or AttackConfig(**config_kwargs)
        
        # Precompute query embeddings
        self._cached_query_embeddings: Optional[np.ndarray] = None

    @property
    def query_embeddings(self) -> np.ndarray:
        """Lazy-load and cache query embeddings."""
        if self._cached_query_embeddings is None:
            logger.info(f"Computing embeddings for {len(self.target_queries)} queries")
            self._cached_query_embeddings = np.array([
                self.embedding_model.embed_query(query) for query in self.target_queries
            ])
        return self._cached_query_embeddings

    async def _embed_async(self, text: str) -> np.ndarray:
        """Async embedding wrapper."""
        embedding_result = await asyncio.to_thread(
            self.embedding_model.embed_query, text
        )
        return np.array(embedding_result)

    async def _generate_tools(
        self, sampled_queries: List[str]
    ) -> List[ToolCandidate]:
        """Generate and score tools for query sample."""
        generation_prompt = f"""Generate {self.attack_config.num_tools_per_sample} tools relevant to ALL these queries:

{chr(10).join(f"- {query}" for query in sampled_queries)}

Each tool must:
- Work across all queries (not query-specific)
- Have a 1-2 word name (lowercase, underscores only)
- Include exact phrases from the queries in its description
"""

        structured_llm = self.llm_optimizer.with_structured_output(GeneratedTools)
        
        # Try async invoke, fallback to sync
        try:
            generation_result = await (
                structured_llm.ainvoke(generation_prompt)
                if hasattr(structured_llm, "ainvoke")
                else asyncio.to_thread(structured_llm.invoke, generation_prompt)
            )
        except Exception as llm_error:
            logger.warning(f"LLM invocation failed: {llm_error}")
            return []

        # Embed queries in sample
        sample_query_embeddings = np.array([
            await self._embed_async(query) for query in sampled_queries
        ])

        # Score and embed tools
        tool_candidates = []
        for generated_tool in generation_result.tools:
            tool_embedding = await self._embed_async(generated_tool.description)
            distances_to_queries = np.array([
                cosine_distance(tool_embedding, q) for q in sample_query_embeddings
            ])
            minimum_distance = float(np.min(distances_to_queries))
            
            tool_candidates.append(ToolCandidate(
                name=generated_tool.name,
                description=generated_tool.description,
                min_distance_to_queries=minimum_distance,
                embedding_vector=tool_embedding,
            ))

        return sorted(tool_candidates, key=lambda candidate: candidate.min_distance_to_queries)

    async def _phase1_iteration(
        self, iteration_number: int, concurrency_semaphore: asyncio.Semaphore
    ) -> List[ToolCandidate]:
        """Single Phase 1 iteration: sample queries and generate tools."""
        async with concurrency_semaphore:
            # Sample queries
            num_queries_to_sample = min(
                self.attack_config.query_sample_size, 
                len(self.target_queries)
            )
            sampled_indices = random.sample(
                range(len(self.target_queries)), 
                num_queries_to_sample
            )
            sampled_queries = [self.target_queries[idx] for idx in sampled_indices]
            
            # logger.debug(
            #     f"Iter {iteration_number}: Sampled {len(sampled_queries)} queries, "
            #     f"first: '{sampled_queries[0][:60]}...'"
            # )

            # Generate tools
            generated_candidates = await self._generate_tools(sampled_queries)
            # logger.info(
            #     f"Iter {iteration_number}: Generated {len(generated_candidates)} tools, "
            #     f"best distance: {generated_candidates[0].min_distance_to_queries:.4f}"
            #     if generated_candidates 
            #     else f"Iter {iteration_number}: No tools generated"
            # )
            
            return generated_candidates

    async def _run_phase1(self) -> List[ToolCandidate]:
        """Phase 1: Generate tool pool via parallel sampling."""
        logger.info(
            f"Phase 1: {self.attack_config.max_generation_iterations} iterations, "
            f"up to {self.attack_config.max_concurrent_tasks} concurrent"
        )
        
        concurrency_semaphore = asyncio.Semaphore(self.attack_config.max_concurrent_tasks)
        generation_tasks = [
            self._phase1_iteration(iteration_num, concurrency_semaphore)
            for iteration_num in range(1, self.attack_config.max_generation_iterations + 1)
        ]
        
        all_generated_tools = []
        for completed_task in tqdm.as_completed(
            generation_tasks, 
            total=len(generation_tasks), 
            desc="Phase 1"
        ):
            all_generated_tools.extend(await completed_task)
        
        logger.success(f"Phase 1: Generated {len(all_generated_tools)} tools")
        return all_generated_tools

    def _find_best_tool(
        self,
        candidate_pool: List[ToolCandidate],
        uncovered_query_indices: List[int],
        query_coverage_counts: Dict[int, int],
    ) -> Optional[tuple[ToolCandidate, List[int]]]:
        """Find tool covering most available queries."""
        best_candidate = None
        best_covered_indices = []
        best_coverage_count = 0

        uncovered_query_embeddings = self.query_embeddings[uncovered_query_indices]
        
        for candidate in candidate_pool:
            # Compute distances to available queries
            distances_to_uncovered = np.array([
                cosine_distance(candidate.embedding_vector, q)
                for q in uncovered_query_embeddings
            ])
            
            # Find covered queries
            newly_covered_indices = [
                uncovered_query_indices[local_idx]
                for local_idx, distance in enumerate(distances_to_uncovered)
                if distance <= self.attack_config.max_embedding_distance
                and query_coverage_counts[uncovered_query_indices[local_idx]] < self.attack_config.num_tools_per_query
            ]
            
            num_newly_covered = len(newly_covered_indices)
            if num_newly_covered == 0:
                continue
            
            # Update best (max coverage, tie-break by distance)
            is_better_coverage = num_newly_covered > best_coverage_count
            is_equal_coverage_better_distance = (
                num_newly_covered == best_coverage_count 
                and candidate.min_distance_to_queries < best_candidate.min_distance_to_queries
            )
            
            if is_better_coverage or is_equal_coverage_better_distance:
                best_candidate = candidate
                best_covered_indices = newly_covered_indices
                best_coverage_count = num_newly_covered

        return (best_candidate, best_covered_indices) if best_candidate else None

    def _run_phase2(
        self, candidate_pool: List[ToolCandidate]
    ) -> tuple[List[Tool], Dict[str, Any]]:
        """Phase 2: Greedy tool selection."""
        logger.info("Phase 2: Greedy selection")
        
        query_coverage_counts = {
            query_idx: 0 for query_idx in range(len(self.target_queries))
        }
        selected_tools = []
        selection_iterations = []
        
        while candidate_pool:
            # Check stopping conditions
            uncovered_query_indices = [
                query_idx for query_idx in range(len(self.target_queries))
                if query_coverage_counts[query_idx] < self.attack_config.num_tools_per_query
            ]
            
            if not uncovered_query_indices:
                logger.info("All queries covered")
                break
            
            if self.attack_config.total_tool_budget and len(selected_tools) >= self.attack_config.total_tool_budget:
                logger.info("Budget reached")
                break
            
            # Select best tool
            selection_result = self._find_best_tool(
                candidate_pool, uncovered_query_indices, query_coverage_counts
            )
            if not selection_result:
                logger.info("No more covering tools")
                break
            
            selected_candidate, covered_query_indices = selection_result
            
            # Update coverage
            for query_idx in covered_query_indices:
                query_coverage_counts[query_idx] += 1
            
            # Remove tool from pool
            candidate_pool = [
                candidate for candidate in candidate_pool 
                if candidate is not selected_candidate
            ]
            
            # Create tool
            tool_identifier = f"attacker_tool_{len(selected_tools) + 1}"
            selected_tools.append(Tool(
                tool_id=tool_identifier,
                name=selected_candidate.name,
                description=selected_candidate.description,
            ))
            
            # Log progress
            num_fully_covered_queries = sum(
                1 for coverage_count in query_coverage_counts.values() 
                if coverage_count >= self.attack_config.num_tools_per_query
            )
            selection_iterations.append({
                "tool_id": tool_identifier,
                "coverage": len(covered_query_indices),
                "queries_covered": num_fully_covered_queries,
                "pool_remaining": len(candidate_pool),
            })
            
            if len(selected_tools) % 10 == 0 or len(covered_query_indices) > 5:
                logger.info(
                    f"Selected {len(selected_tools)} tools, "
                    f"{num_fully_covered_queries}/{len(self.target_queries)} queries fully covered"
                )
        
        logger.success(f"Phase 2: Selected {len(selected_tools)} tools")
        
        num_fully_covered_queries = sum(
            1 for coverage_count in query_coverage_counts.values() 
            if coverage_count >= self.attack_config.num_tools_per_query
        )
        
        return selected_tools, {
            "iterations": selection_iterations,
            "total_tools": len(selected_tools),
            "queries_covered": num_fully_covered_queries,
        }

    def attack(self) -> tuple[List[Tool], Dict[str, Any]]:
        """
        Execute ToolFlood attack.
        
        Returns:
            (selected_tools, attack_results) where attack_results contains phase statistics
        """
        logger.info(
            f"ToolFlood: {len(self.target_queries)} queries, "
            f"{self.attack_config.num_tools_per_query} tools/query, "
            f"threshold={self.attack_config.max_embedding_distance}"
        )
        
        # Phase 1: Generate candidates
        candidate_pool = asyncio.run(self._run_phase1())
        
        # Phase 2: Select tools
        selected_tools, phase2_results = self._run_phase2(candidate_pool)
        
        attack_results = {
            "phase1": {"generated": len(candidate_pool)},
            "phase2": phase2_results,
            "config": {
                "num_tools_per_query": self.attack_config.num_tools_per_query,
                "max_embedding_distance": self.attack_config.max_embedding_distance,
                "total_tool_budget": self.attack_config.total_tool_budget,
            },
        }
        
        return selected_tools, attack_results