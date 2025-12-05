# Benchmarks: AutoEnv-36 Usage

This folder contains the 36 standalone AutoEnv environments. Each environment ships with levels, config, code, validator, and docs ready for evaluation.

## Layout
- `*/`: Individual environment folders (with `levels/` and optional `val_levels/`)
- `base/`: Benchmark runner and built-in solver
- `run.py`: CLI entrypoint
- `api.py`: Programmatic interfaces
- `config/benchmark/bench_llm_example.yaml`: Example config

## Quick Start (built-in SolverAgent with cost tracking)
1) Provide model creds in `config/model_config.yaml` (or via env vars).
2) Run:
```bash
python benchmarks/run.py \
  --config config/benchmark/bench_llm_example.yaml \
  --mode test \
  --max-worlds 5
```
Args:
- `--mode {test,val}`: choose `levels/` vs `val_levels/`
- `--max-worlds N`: limit worlds per env (default from config)

## Custom Agent (scores only, no cost)
Your agent must expose `run(env, env_info)` (sync or async). `env_info` includes `world_id`, `agent_instruction`, `action_space`, `max_step`.
```bash
python benchmarks/run.py \
  --agent your_module:YourAgentAttr \
  --agent-kwargs '{"foo": 1}' \
  --mode val
```
`--agent` accepts `module:Attr` or `/path/to/file.py:Attr`; Attr can be a class, factory function, or a pre-built instance. `--agent-kwargs` is unpacked into the constructor/factory call.

## Programmatic use
```python
import asyncio
from benchmarks.api import benchmark_llms, benchmark_custom_agent

async def run_llm():
    await benchmark_llms(
        env_paths=["benchmarks/20_GridNavigation"],
        llm_names=["your_llm_name"],
        world_mode="test",
        max_worlds=5,
    )

async def run_custom():
    from your_module import YourAgent
    await benchmark_custom_agent(
        env_paths=["benchmarks/20_GridNavigation"],
        agent_factory=YourAgent,
        agent_kwargs={"foo": 1},
        world_mode="val",
    )

asyncio.run(run_llm())
```

## Outputs
- Results CSV: `workspace/logs/results/<timestamp>_env_result.csv`
- Trajectories: `workspace/logs/trajectories/`
- Cost: recorded only for the LLM branch (based on model usage summary)
