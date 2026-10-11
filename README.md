# AHD-RL

Code for the paper **Dynamic Task Allocation and Scheduling for Heterogeneous Robot Teams via Attention-Based Reinforcement Learning**.

Attention-based Hierarchical Decision Reinforcement Learning (AHD-RL) constructs schedules for cooperative tasks performed by heterogeneous robot teams. The policy selects a task from the current team state and forms its execution coalition through successive agent decisions. Bidirectional attention represents task and agent information, and schedule simulation provides the state for subsequent decisions.

This repository contains AHD-RL, its three ablation variants, and the RCA-RL, TACO, SAS, and SLPSO comparison methods. Model weights for AHD-RL, the ablations, and RCA-RL are included.

## Code structure

- `src/ahd_rl_mrta/proposed/ahd_rl/`: instance generation, scheduling environment, policy network, ablation variants, inference, and training.
- `src/ahd_rl_mrta/baselines/`: RCA-RL, TACO, SAS, and SLPSO.
- `src/ahd_rl_mrta/models/`: model weights for the learned methods.
- `examples/instances/`: three G01 instances (30 tasks, 5 agents) from the original S-EC, S-HC, and D-HC test sets.
- `src/ahd_rl_mrta/solve.py` and `src/ahd_rl_mrta/__main__.py`: Python and command-line interfaces.

## Running instructions

Install the package from the repository root:

```bash
python -m pip install -e .
```

Run a method on one generated instance:

```bash
python -m ahd_rl_mrta --method ahd-rl --group G01 --type 1 --seed 123
python -m ahd_rl_mrta --method ahd-rl-nca --group G01 --type 1 --seed 123
python -m ahd_rl_mrta --method rca-rl --group G01 --type 1 --seed 123
python -m ahd_rl_mrta --method taco --group G01 --type 1 --seed 123 --max-evals 1000
```

Run a supplied test instance with any method:

```bash
python -m ahd_rl_mrta --method rca-rl --instance examples/instances/g01_s_ec.npz
```

The three example files are the first instances in the original G01 S-EC, S-HC, and D-HC test groups. They contain instance data only.

The `--method` options are `ahd-rl`, `ahd-rl-nca`, `ahd-rl-nrt`, `ahd-rl-ncr`, `rca-rl`, `taco`, `sas`, and `slpso`. Group numbers `1`–`13` directly correspond to the paper labels `G01`–`G13`. Instance types `1`, `2`, and `3` correspond to S-EC, S-HC, and D-HC. For learned methods, `--samples 10` evaluates ten sampled schedules for the same instance and returns the best feasible schedule.

The Python interface accepts an instance in memory:

```python
from ahd_rl_mrta.proposed.ahd_rl.environment import generate_instance_by_type
from ahd_rl_mrta.solve import solve

instance = generate_instance_by_type(1, "1", seed=123)  # G01
result = solve(instance, method="ahd-rl", seed=123)
print(result["makespan"])
```

Python 3.7 or newer, PyTorch, NumPy, and SciPy are required. The models can run on a CPU.

## Training

Train AHD-RL from scratch on a small group:

```bash
python -m ahd_rl_mrta.proposed.ahd_rl.training --groups G01 --instances-per-update 1 --trajectories 2
```

The defaults use four instances and eight sampled trajectories per update. Passing multiple groups trains in small-, medium-, then large-instance order. Only the final model weights are written to `output/`.

The RCA-RL code is adapted from [HeteroMRTA](https://github.com/marmotlab/HeteroMRTA); see its [source and license notes](src/ahd_rl_mrta/baselines/rca_rl/SOURCE.md). This repository uses the [Apache-2.0 License](LICENSE).
