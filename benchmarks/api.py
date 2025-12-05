"""
Lightweight benchmark interfaces for AutoEnv-36 environments.

Two entry points:
1) benchmark_llms: run provided SolverAgent with different LLM backends, record score + cost
2) benchmark_custom_agent: plug in any agent that exposes run(env, env_info) and record scores
"""

import asyncio
import os
import re
import time
from typing import Any, Callable, Iterable, Optional, Sequence

from benchmarks.base.agent import SolverAgent
from benchmarks.base.benchmark import Benchmark
from base.engine.async_llm import LLMsConfig, create_llm_instance


def _safe_llm_label(llm: Any, name: str) -> str:
    """Generate a readable label for an LLM instance."""
    if hasattr(llm, "config") and hasattr(llm.config, "model"):
        return str(getattr(llm.config, "model"))
    return str(name)


def _default_cost_meter(llm: Any) -> Callable[[], float]:
    """Return a callable that reads cumulative cost from an AsyncLLM-like object."""
    return lambda: float(llm.get_usage_summary().get("total_cost", 0.0))


async def benchmark_llms(
    env_paths: Sequence[str],
    llm_names: Iterable[str],
    *,
    world_mode: str = "test",
    world_concurrency: int = 1,
    max_worlds: Optional[int] = 5,
    result_folder_path: str = "workspace/logs/results",
    trajectory_folder_path: str = "workspace/logs/trajectories",
    solver_cls: Callable[..., Any] = SolverAgent,
) -> list[Benchmark]:
    """
    Run benchmarks with the built-in SolverAgent across multiple LLMs.

    Args:
        env_paths: Paths to environment folders.
        llm_names: Iterable of LLM names configured in model_config.yaml.
        world_mode: "test" or "val" to choose levels/val_levels.
        world_concurrency: Concurrent worlds per env.
        max_worlds: Limit number of worlds per env (None for all).
        result_folder_path: Where CSV summaries are written.
        trajectory_folder_path: Where trajectories are saved.
        solver_cls: Agent class/factory to use (defaults to SolverAgent).

    Returns:
        List of Benchmark instances with populated results/costs.
    """
    timestamp = time.strftime("%m%d_%H%M")
    benches: list[Benchmark] = []

    for llm_name in llm_names:
        llm = create_llm_instance(LLMsConfig.default().get(llm_name))
        llm_label = _safe_llm_label(llm, llm_name)

        for env_path in env_paths:
            env_name = os.path.basename(env_path.rstrip("/"))
            llm_dir = re.sub(r"[^A-Za-z0-9_.-]+", "-", llm_label)
            trajectory_folder = f"{trajectory_folder_path}/{timestamp}_{env_name}_{llm_dir}"

            bench = Benchmark(
                env_folder_path=env_path,
                result_folder_path=result_folder_path,
                trajectory_folder_path=trajectory_folder,
                llm_name=str(llm_label),
                timestamp=timestamp,
                env_name=env_name,
            )
            benches.append(bench)

            await bench.execute(
                solver_factory=solver_cls,
                solver_kwargs={
                    "llm": llm,
                    "trajectory_folder_path": trajectory_folder,
                    "unique_run_id": f"{timestamp}_{llm_label}",
                },
                world_concurrency=world_concurrency,
                max_worlds=max_worlds,
                world_mode=world_mode,
                cost_meter=_default_cost_meter(llm),
            )

    return benches


async def benchmark_custom_agent(
    env_paths: Sequence[str],
    agent_factory: Any,
    *,
    agent_kwargs: Optional[dict] = None,
    world_mode: str = "test",
    world_concurrency: int = 1,
    max_worlds: Optional[int] = None,
    result_folder_path: str = "workspace/logs/results",
    trajectory_folder_path: str = "workspace/logs/trajectories",
    agent_name: str = "custom_agent",
) -> list[Benchmark]:
    """
    Run benchmarks with any custom agent. Agent must expose run(env, env_info).
    Cost is not tracked (caller can manage their own).
    """
    agent_kwargs = agent_kwargs or {}
    timestamp = time.strftime("%m%d_%H%M")
    benches: list[Benchmark] = []
    tasks = []

    for env_path in env_paths:
        env_name = os.path.basename(env_path.rstrip("/"))
        trajectory_folder = f"{trajectory_folder_path}/{timestamp}_{env_name}_{agent_name}"

        bench = Benchmark(
            env_folder_path=env_path,
            result_folder_path=result_folder_path,
            trajectory_folder_path=trajectory_folder,
            llm_name=agent_name,
            timestamp=timestamp,
            env_name=env_name,
        )
        benches.append(bench)

        tasks.append(
            bench.execute(
                solver_factory=agent_factory,
                solver_kwargs=agent_kwargs,
                world_concurrency=world_concurrency,
                max_worlds=max_worlds,
                world_mode=world_mode,
                cost_meter=lambda: None,
            )
        )

    await asyncio.gather(*tasks)
    return benches
