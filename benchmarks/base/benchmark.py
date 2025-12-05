import os
import time
import importlib.util
import sys
import yaml
import asyncio
import json
import inspect
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Callable
from base.engine.utils import read_file_content
from base.engine.logs import logger


class EnvWrapper:
    """Environment wrapper to ensure operations run under the env directory."""
    
    def __init__(self, env, env_folder_path):
        self._env = env
        self._env_folder_path = env_folder_path
        self._original_cwd = os.getcwd()
    
    def __getattr__(self, name):
        """Proxy attribute access to the wrapped environment."""
        attr = getattr(self._env, name)
        
        # If callable, run under the environment folder for correct relative paths
        if callable(attr):
            def wrapped_method(*args, **kwargs):
                original_cwd = os.getcwd()
                try:
                    os.chdir(self._env_folder_path)
                    return attr(*args, **kwargs)
                finally:
                    os.chdir(original_cwd)
            return wrapped_method
        else:
            return attr


class Benchmark(BaseModel):
    env_folder_path: str = Field(default="")
    result_folder_path: str = Field(default="workspace/logs/results")
    trajectory_folder_path: str = Field(default="workspace/logs/trajectories")
    llm_name: str = Field(default="")
    timestamp: str = Field(default="")
    env_name: str = Field(default="")
    results: Dict = Field(default_factory=dict)
    max_rewards: Dict = Field(default_factory=dict)
    costs: Dict[str, float] = Field(default_factory=dict)
    per_world_max_rewards: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    world_details: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    env_durations: Dict[str, float] = Field(default_factory=dict)
    event_totals: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    env_world_ids: Dict[str, List[str]] = Field(default_factory=dict)

    def _init_result_folder(self):
        # Keep a single CSV under result_folder_path; do not nest per-LLM folders
        os.makedirs(self.result_folder_path, exist_ok=True)
        # Trajectory folder is controlled by caller; ensure directory exists
        os.makedirs(self.trajectory_folder_path, exist_ok=True)

    def _list_env_worlds(self, env_folder_path: str, mode="test") -> List[str]:
        """List level IDs for the given environment folder."""
        env_path = Path(env_folder_path)
        if mode == "val":
            levels_dir = env_path / "val_levels"
        else:
            levels_dir = env_path / "levels"
        
        if not levels_dir.exists():
            logger.warning(f"Level directory does not exist: {levels_dir}")
            return []
        
        levels = []
        for level_file in levels_dir.glob("*.yaml"):
            level_id = level_file.stem
            levels.append(level_id)
        
        return sorted(levels)

    def _load_env(self, env_folder_path: str):
        """
        Dynamically load the environment class from env_main.py that subclasses SkinEnv.
        """
        env_path = Path(env_folder_path)
        env_main_path = env_path / "env_main.py"
        
        if not env_main_path.exists():
            raise ValueError(f"env_main.py not found: {env_main_path}")
        
        # Save current working directory
        original_cwd = os.getcwd()
        
        # Dynamic import
        spec = importlib.util.spec_from_file_location("env_main", str(env_main_path))
        env_module = importlib.util.module_from_spec(spec)
        
        # Add env directory to sys.path for relative imports
        env_dir_str = str(env_path.resolve())
        if env_dir_str not in sys.path:
            sys.path.insert(0, env_dir_str)
        
        try:
            # Temporarily cd into env dir to support relative paths
            os.chdir(env_dir_str)
            spec.loader.exec_module(env_module)
        except Exception as e:
            raise ImportError(f"Failed to load env module {env_folder_path}: {e}")
        finally:
            # Restore original cwd
            os.chdir(original_cwd)
            # Clean sys.path
            if env_dir_str in sys.path:
                sys.path.remove(env_dir_str)
        
        # Locate subclass of SkinEnv
        from base.env.base_env import SkinEnv
        
        env_class = None
        for attr_name in dir(env_module):
            attr = getattr(env_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, SkinEnv) and 
                attr != SkinEnv):
                env_class = attr
                break
        
        if env_class is None:
            raise ValueError(f"No valid SkinEnv subclass found in {env_folder_path}")
        
        return env_class

    def _validate_level(self, env_folder_path: str, level_id: str) -> bool:
        """Validate that a level file exists and is loadable (supports val_levels and levels)."""
        env_path = Path(env_folder_path)
        # Prefer validation in both val_levels and levels to match listing behavior
        candidates = [
            env_path / "val_levels" / f"{level_id}.yaml",
            env_path / "levels" / f"{level_id}.yaml",
        ]
        for level_path in candidates:
            if level_path.exists():
                try:
                    with open(level_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    return True
                except yaml.YAMLError:
                    return False
        return False

    def _get_env_info(self, env_folder_path: str) -> Dict[str, Any]:
        """Read environment metadata (instruction, action space, config)."""
        env_path = Path(env_folder_path)
        
        info = {}
        
        # Agent instruction
        agent_instruction_path = env_path / "agent_instruction.txt"
        if agent_instruction_path.exists():
            info["agent_instruction"] = read_file_content(str(agent_instruction_path))
        
        # Action space
        action_space_path = env_path / "action_space.txt"
        if action_space_path.exists():
            info["action_space"] = read_file_content(str(action_space_path))
        
        # Config
        config_path = env_path / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                info["config"] = config
        
        return info
    
    def _create_solver(self, solver_factory: Any, solver_kwargs: Dict[str, Any]):
        """
        Create solver instance from a class, factory callable, or pre-built instance.

        - If solver_factory is a class, it will be instantiated with **solver_kwargs
        - If solver_factory is callable, it will be invoked with **solver_kwargs
        - Otherwise, the factory itself is treated as the solver instance (kwargs ignored)
        """
        if inspect.isclass(solver_factory):
            return solver_factory(**solver_kwargs)
        if callable(solver_factory):
            return solver_factory(**solver_kwargs)
        return solver_factory
    
    async def _run_solver(self, solver: Any, env: Any, env_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run solver.run (async or sync)."""
        if not hasattr(solver, "run"):
            raise AttributeError("solver must provide a 'run(env, env_info)' method")
        result = solver.run(env, env_info)
        if inspect.isawaitable(result):
            return await result
        return result

    def _load_env_world(self, env_folder_path: str, world_id: str):
        """Load environment instance and world info."""
        # Validate level
        if not self._validate_level(env_folder_path, world_id):
            raise ValueError(f"Level {world_id} in environment {env_folder_path} is missing or invalid")
        
        # Map world_id to ../val_levels/ when env expects ./levels
        from pathlib import Path as _Path
        _env_p = _Path(env_folder_path)
        levels_path = _env_p / "levels" / f"{world_id}.yaml"
        val_levels_path = _env_p / "val_levels" / f"{world_id}.yaml"
        effective_world_id = world_id
        if (not levels_path.exists()) and val_levels_path.exists():
            # The env uses ./levels/{world_id}.yaml internally; redirect to val_levels
            effective_world_id = f"../val_levels/{world_id}"

        # Gather env info
        env_info_data = self._get_env_info(env_folder_path)
        if not env_info_data.get("agent_instruction") or not env_info_data.get("action_space"):
            raise ValueError(f"Environment {env_folder_path} missing required files")
        
        # Prepare env_info
        env_info = {
            "world_id": effective_world_id,
            "agent_instruction": env_info_data["agent_instruction"],
            "action_space": env_info_data["action_space"],
        }
        
        # Extract max_steps if present
        if "config" in env_info_data:
            config = env_info_data["config"]
            if isinstance(config, dict):
                termination = config.get("termination", {})
                if isinstance(termination, dict) and "max_steps" in termination:
                    env_info["max_step"] = termination["max_steps"]
        
        # Load env class
        env_class = self._load_env(env_folder_path)
        
        # Instantiate under env directory to honor relative paths
        env_name = Path(env_folder_path).name
        original_cwd = os.getcwd()
        try:
            os.chdir(env_folder_path)
            env = env_class(env_id=f"{env_name}_benchmark")
        finally:
            os.chdir(original_cwd)
        
        # Wrap to enforce cwd during calls
        wrapped_env = EnvWrapper(env, env_folder_path)
        
        return wrapped_env, env_info

    def _load_max_rewards(self, env_folder_path: str) -> Dict[str, float]:
        """Load per-level maximum rewards."""
        env_path = Path(env_folder_path)
        max_rewards_path = env_path / "level_max_rewards.json"
        
        if not max_rewards_path.exists():
            logger.warning(f"Max reward file not found: {max_rewards_path}")
            return {}
        
        try:
            with open(max_rewards_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract max reward per level
            max_rewards = {}
            levels_data = data.get("levels", {})
            for level_name, level_info in levels_data.items():
                # Strip .yaml suffix
                level_id = level_name.replace('.yaml', '')
                max_reward = level_info.get("max_reward", 0.0)
                max_rewards[level_id] = max_reward
            
            return max_rewards
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to read max reward file {max_rewards_path}: {e}")
            return {}

    def _calculate_max_reward_total(self, env_folder_path: str, world_ids: List[str]) -> float:
        """Compute total max reward for selected worlds."""
        max_rewards_dict = self._load_max_rewards(env_folder_path)

        filtered_max_rewards = {world_id: max_rewards_dict.get(world_id, 0.0) for world_id in world_ids}
        self.per_world_max_rewards[env_folder_path] = filtered_max_rewards

        total_max_reward = sum(filtered_max_rewards.values())
        return total_max_reward


    def _save_env_result_to_csv(self):
        """Append a single row to timestamped result CSV safely for multi-process.

        Writes header if file does not exist. Avoids read/modify/write races.
        """
        import csv
        import shutil
        import tempfile

        timestamp_prefix = self.timestamp or time.strftime("%m%d_%H%M")
        env_name = self.env_name or "unknown_env"
        csv_filename = f"{timestamp_prefix}_{env_name}_result.csv"
        csv_path = os.path.join(self.result_folder_path, csv_filename)

        if not self.results:
            return

        env_path = list(self.results.keys())[-1]
        total_reward = float(self.results.get(env_path, 0.0) or 0.0)
        max_reward_total = float(self.max_rewards.get(env_path, 0.0) or 0.0)
        ratio = (total_reward / max_reward_total) if max_reward_total else None
        cost = self.costs.get(env_path)
        world_details = self.world_details.get(env_path, [])
        events_summary = self.event_totals.get(env_path, {}) or {}
        duration_seconds = self.env_durations.get(env_path)
        loaded_world_ids = [str(world_id) for world_id in self.env_world_ids.get(env_path, [])]
        if loaded_world_ids:
            world_ids = loaded_world_ids
        else:
            world_ids = [str(detail.get("world_id")) for detail in world_details if detail.get("world_id")]
        world_count = len(world_ids)
        total_steps = sum(int(detail.get("steps") or 0) for detail in world_details)
        avg_steps_per_world = (total_steps / world_count) if world_count else None
        avg_reward_per_world = (total_reward / world_count) if world_count else None
        ratio_values = [float(detail.get("ratio")) for detail in world_details if detail.get("ratio") is not None]
        success_worlds = sum(1 for value in ratio_values if value >= 0.999)

        row = {
            "env_folder_path": env_path,
            "llm": self.llm_name or "",
            "total_reward": total_reward,
            "max_reward_total": max_reward_total,
            "ratio": ratio,
            "cost": cost,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "world_count": world_count,
            "world_ids": "|".join(world_ids),
            "avg_reward_per_world": avg_reward_per_world,
            "total_steps": total_steps,
            "avg_steps_per_world": avg_steps_per_world,
            "success_worlds": success_worlds,
            "events_summary": json.dumps(events_summary, ensure_ascii=False) if events_summary else "",
            "duration_seconds": duration_seconds,
        }

        os.makedirs(self.result_folder_path, exist_ok=True)

        exists = os.path.exists(csv_path)
        try:
            import fcntl  # type: ignore
        except Exception:
            fcntl = None

        if exists and os.path.getsize(csv_path) > 0:
            try:
                with open(csv_path, "r", encoding="utf-8", newline="") as rf:
                    reader = csv.DictReader(rf)
                    existing_fieldnames = reader.fieldnames or []
                    old_rows = list(reader)
                if existing_fieldnames != list(row.keys()):
                    fd, tmp_path = tempfile.mkstemp(prefix="bench_migrate_", suffix=".csv")
                    os.close(fd)
                    with open(tmp_path, "w", encoding="utf-8", newline="") as wf:
                        writer = csv.DictWriter(wf, fieldnames=list(row.keys()))
                        writer.writeheader()
                        for old_row in old_rows:
                            new_row = {key: old_row.get(key, "") for key in row.keys()}
                            writer.writerow(new_row)
                    shutil.move(tmp_path, csv_path)
                    exists = os.path.exists(csv_path)
            except Exception:
                pass

        fieldnames = list(row.keys())

        with open(csv_path, "a+", newline="", encoding="utf-8") as f:
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists or os.path.getsize(csv_path) == 0:
                writer.writeheader()
            writer.writerow(row)
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


    async def execute(
        self,
        solver_factory: Any,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        world_concurrency: int = 1,
        max_worlds: Optional[int] = None,
        world_mode: str = "test",
        cost_meter: Optional[Callable[[], float]] = None,
    ):
        """
        Run the benchmark.
        Args:
            solver_factory: Agent class/callable/instance providing run(env, env_info)
            solver_kwargs: kwargs to build the solver (e.g., llm)
            world_concurrency: concurrent worlds per env
            max_worlds: limit number of worlds (None = all)
            world_mode: source of levels, "test" uses levels/, "val" uses val_levels/
            cost_meter: optional callable returning cumulative cost for delta
        """
        solver_kwargs = solver_kwargs or {}

        # Initialize result folders
        self._init_result_folder()

        env_folder_path = self.env_folder_path

        # Default cost_meter: if solver_kwargs has llm, try usage summary
        if cost_meter is None:
            llm = solver_kwargs.get("llm")
            if llm is not None and hasattr(llm, "get_usage_summary"):
                def _default_meter():
                    return float(llm.get_usage_summary().get("total_cost", 0.0))
                cost_meter = _default_meter

        prev_cost = None
        if cost_meter is not None:
            try:
                prev_cost = float(cost_meter())
            except Exception:
                prev_cost = None

        # Select world source by mode
        mode = str(world_mode or "test").lower()
        if mode not in {"test", "val"}:
            mode = "test"
        world_ids = self._list_env_worlds(env_folder_path, mode=mode)
        if max_worlds is not None:
            world_ids = world_ids[:max_worlds]
        self.env_world_ids[env_folder_path] = list(world_ids)

        max_reward_total = self._calculate_max_reward_total(env_folder_path, world_ids)
        self.max_rewards[env_folder_path] = max_reward_total

        start_time = time.perf_counter()

        async def run_world(world_id: str):
            solver = self._create_solver(solver_factory, solver_kwargs)
            env, env_info = self._load_env_world(env_folder_path, world_id)
            result = await self._run_solver(solver, env, env_info)
            return world_id, result, env_info

        semaphore = asyncio.Semaphore(world_concurrency)

        async def run_world_with_semaphore(world_id: str):
            async with semaphore:
                return await run_world(world_id)

        raw_results = await asyncio.gather(*[
            run_world_with_semaphore(world_id) for world_id in world_ids
        ])

        elapsed = time.perf_counter() - start_time
        self.env_durations[env_folder_path] = elapsed

        per_world_max = self.per_world_max_rewards.get(env_folder_path, {})
        aggregated_events: Dict[str, int] = defaultdict(int)
        world_details: List[Dict[str, Any]] = []
        world_rewards: List[float] = []

        for world_id, result, env_info in raw_results:
            reward = float(result.get("total_reward") or 0.0)
            world_rewards.append(reward)
            steps = int(result.get("step") or 0)
            events_count = result.get("events_count") or {}
            if isinstance(events_count, dict):
                for event_name, count in events_count.items():
                    try:
                        aggregated_events[event_name] += int(count)
                    except Exception:
                        continue
            max_reward_world = per_world_max.get(world_id, 0.0)
            ratio_world = (reward / max_reward_world) if max_reward_world else None
            world_details.append(
                {
                    "world_id": world_id,
                    "reward": reward,
                    "steps": steps,
                    "events_count": events_count,
                    "max_reward": max_reward_world,
                    "ratio": ratio_world,
                    "max_step": env_info.get("max_step"),
                }
            )

        self.world_details[env_folder_path] = world_details
        self.event_totals[env_folder_path] = dict(aggregated_events)

        cur_env_total_reward = sum(world_rewards)
        self.results[env_folder_path] = cur_env_total_reward
        env_cost = None
        if cost_meter is not None and prev_cost is not None:
            try:
                env_cost = max(0.0, float(cost_meter()) - prev_cost)
            except Exception:
                env_cost = None
        self.costs[env_folder_path] = env_cost
        self._save_env_result_to_csv()

        env_desc = f"Environment {env_folder_path}" + (f" (first {max_worlds} worlds)" if max_worlds else "")
        logger.info(f"{env_desc}:")
        logger.info(f"  Total reward: {cur_env_total_reward}")
        logger.info(f"  Max reward: {max_reward_total}")
        logger.info(f"  Success ratio: {cur_env_total_reward/max_reward_total*100:.2f}%" if max_reward_total > 0 else "  Success ratio: N/A")

        logger.info(f"Benchmark Results Saved to {self.result_folder_path}")


    async def execute_with_five(
        self,
        solver_factory: Any,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        world_concurrency: int = 1,
        world_mode: str = "test",
        cost_meter: Optional[Callable[[], float]] = None,
    ):
        """
        Convenience helper to run up to 5 levels.
        Args:
            solver_factory: Agent class/callable/instance providing run(env, env_info)
            solver_kwargs: kwargs to build the solver
            world_concurrency: concurrent worlds per env
            world_mode: levels source, "test" uses levels/, "val" uses val_levels/
            cost_meter: Optional cost reader to compute delta cost
        """
        await self.execute(
            solver_factory,
            solver_kwargs,
            world_concurrency,
            max_worlds=5,
            world_mode=world_mode,
            cost_meter=cost_meter,
        )


    async def __call__(self, solver_factory: Any, solver_kwargs: Optional[Dict[str, Any]] = None):
        """Convenience callable interface."""
        await self.execute(solver_factory, solver_kwargs)
