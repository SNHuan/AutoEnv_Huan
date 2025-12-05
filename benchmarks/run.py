import os
import time
import yaml
import argparse
import asyncio
import re
import json
import importlib
import importlib.util
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.base.benchmark import Benchmark
from benchmarks.base.agent import SolverAgent
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger


def _load_custom_agent(target: str):
    """
    Load custom agent from "module:Attr" or "/path/to/file.py:Attr".
    Attr can be a class, factory callable, or a pre-built instance.
    """
    if ":" not in target:
        raise ValueError("agent must be in 'module:ClassName' format")
    module_part, class_name = target.split(":", 1)
    module = None

    # Support direct .py path
    module_path = Path(module_part)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location("custom_agent_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from path: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)

    agent_cls = getattr(module, class_name, None)
    if agent_cls is None:
        raise ImportError(f"Class {class_name} not found in {module_part}")
    # If Attr is already an instance, return directly (kwargs will be ignored downstream)
    if not callable(agent_cls):
        return agent_cls
    return agent_cls


async def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on AutoEnv-36")
    parser.add_argument("--config", type=str, default="config/benchmark/bench_llm_example.yaml", help="Path to benchmark configuration YAML file")
    parser.add_argument("--mode", choices=["test", "val"], default=None, help="Use test levels or validation levels")
    parser.add_argument("--max-worlds", type=int, default=None, help="Max number of worlds to run per env (default: config or 5 for LLM mode)")
    parser.add_argument("--agent", type=str, default=None, help="Custom agent path in form module:Attr (class/factory/instance; sync or async run supported)")
    parser.add_argument("--agent-kwargs", type=str, default=None, help="JSON string of kwargs passed to the custom agent")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env_paths = config.get("env_folder_paths", [])
    result_root = config.get("result_folder_path", "workspace/logs/results")
    trajectory_root = config.get("trajectory_folder_path", "workspace/logs/trajectories")
    world_concurrency = config.get("world_concurrency", 1)
    default_max_worlds = config.get("max_worlds", 5)
    max_worlds = args.max_worlds if args.max_worlds is not None else default_max_worlds
    world_mode = (args.mode or config.get("world_mode") or "test").lower()
    if world_mode not in {"test", "val"}:
        world_mode = "test"

    # Execute benchmark for each LLM and environment
    timestamp = time.strftime("%m%d_%H%M")

    # Branch 1: custom agent (self-contained, no cost)
    custom_tasks = []
    if args.agent:
        agent_kwargs = {}
        if args.agent_kwargs:
            try:
                agent_kwargs = json.loads(args.agent_kwargs)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid --agent-kwargs JSON: {e}") from e
        solver_factory = _load_custom_agent(args.agent)
        for env_path in env_paths:
            env_name = os.path.basename(env_path.rstrip("/"))
            trajectory_folder_path = f"{trajectory_root}/{timestamp}_{env_name}_custom"
            benchmark = Benchmark(
                env_folder_path=env_path,
                result_folder_path=result_root,
                trajectory_folder_path=trajectory_folder_path,
                llm_name="custom_agent",
                timestamp=timestamp,
                env_name=env_name,
            )
            custom_tasks.append(
                benchmark.execute(
                    solver_factory=solver_factory,
                    solver_kwargs=agent_kwargs,
                    world_concurrency=world_concurrency,
                    max_worlds=max_worlds,
                    world_mode=world_mode,
                    cost_meter=lambda: None,  # external agents manage their own cost
                )
            )
        if custom_tasks:
            await asyncio.gather(*custom_tasks)
        logger.info("Benchmark run finished for custom agent.")
        return

    # Branch 2: provided SolverAgent + different LLMs (cost tracked, sequential per LLM for accurate cost)
    llms = {}
    for llm_name in config.get("llms", []):
        llm = create_llm_instance(LLMsConfig.default().get(llm_name))
        llms[llm_name] = llm

        for env_path in env_paths:
            env_name = os.path.basename(env_path.rstrip("/"))
            llm_dir = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(llm_name))
            trajectory_folder_path = f"{trajectory_root}/{timestamp}_{env_name}_{llm_dir}"

            benchmark = Benchmark(
                env_folder_path=env_path,
                result_folder_path=result_root,
                trajectory_folder_path=trajectory_folder_path,
                llm_name=str(llm_name),
                timestamp=timestamp,
                env_name=env_name,
            )

            def _cost_meter(llm=llm):
                return float(llm.get_usage_summary().get("total_cost", 0.0))

            await benchmark.execute(
                solver_factory=SolverAgent,
                solver_kwargs={
                    "llm": llm,
                    "trajectory_folder_path": trajectory_folder_path,
                    "unique_run_id": f"{timestamp}_{llm_name}",
                },
                world_concurrency=world_concurrency,
                max_worlds=max_worlds,
                world_mode=world_mode,
                cost_meter=_cost_meter,
            )
    
    logger.info("Benchmark run finished.")
    for name, instance in llms.items():
        cost = instance.get_usage_summary()['total_cost']
        logger.info(f"{name}: Cost: {cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
