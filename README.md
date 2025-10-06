# Physics-Informed Twin Delayed Deep Deterministic Policy Gradient (PI-TD3) for EV Charging Management

This repository implements **Physics-Informed TD3 (PI-TD3)**, a novel reinforcement learning algorithm that integrates differentiable physics models into the TD3 framework for electric vehicle (EV) charging management in vehicle-to-grid (V2G) systems. The method leverages known environment dynamics to enable gradient propagation through multi-step rollouts, improving sample efficiency and policy performance.

## Proposed Method

Physics-Informed TD3 extends the standard TD3 algorithm by incorporating differentiable transition and reward functions that model the physical dynamics of EV charging systems. Unlike conventional model-free RL approaches, PI-TD3 enables direct gradient propagation from cumulative rewards through K-step rollouts, allowing the policy network to learn more effectively from the underlying physics.

### Key Features

- **Differentiable Physics Models**: Implements smooth, differentiable approximations of state transitions (SoC updates, voltage dynamics) and reward functions (costs, voltage violations)
- **K-step Lookahead**: Performs multi-step rollouts during training using the learned physics models
- **Gradient Propagation**: Enables end-to-end gradient flow from future rewards back to the policy network
- **Sample Efficiency**: Reduces the number of environment interactions required for training

### Method Overview

![Fig. 2: The policy network πθ generates actions that, together with the known environment dynamics (e.g., SoC and voltage updates) and sampled exogenous variables (e.g., loads, prices, and EV arrivals), are used to simulate K-step rollouts through the differentiable transition T(s, a) and reward R(s, a) functions. This enables direct gradient propagation from cumulative rewards back through the rollout.](path/to/figure2.png)

*Figure 2: Physics-Informed rollout mechanism enabling gradient propagation through differentiable environment dynamics.*

### Algorithm

![Algorithm 1 Physics-Informed TD3](path/to/algorithm1.png)

*Algorithm 1: Physics-Informed TD3 training procedure.*

The PI-TD3 algorithm alternates between:
1. Collecting trajectories using the current policy
2. Performing K-step rollouts with differentiable physics models
3. Computing actor gradients through the rollout chain
4. Updating critic networks using TD(λ) targets
5. Soft-updating target networks

## Repository Structure

### Core Directories

- **`algorithms/`**: Reinforcement learning algorithm implementations
  - `pi_TD3.py`: Physics-Informed TD3 algorithm (main contribution)
  - `TD3.py`: Standard TD3 baseline implementation
  - `pi_ppo.py`: Physics-Informed PPO variant
  - `SAC/`: Soft Actor-Critic implementations
  - `shac.py`, `sapo.py`: Hindsight and other baseline methods
  - `utils.py`: Shared utilities for RL algorithms (TD-λ computation, replay buffers)

- **`agent/`**: Environment-specific components
  - `transition_fn.py`: Differentiable state transition functions (SoC updates, voltage dynamics)
  - `loss_fn.py`: Differentiable reward/cost functions (V2G costs, voltage violations)
  - `reward.py`: Environment reward function definitions
  - `state.py`: State representation functions
  - `utils.py`: Replay buffers and trajectory storage

- **`ev2gym/`**: EV2Gym environment implementation
  - Core gymnasium-compatible environment for EV charging simulation
  - Includes grid models, EV profiles, and charging station dynamics

- **`config_files/`**: YAML configuration files for different scenarios
  - `v2g_grid_*.yaml`: Various V2G scenarios with different grid sizes
  - `PST_V2G_*.yaml`: Power system test configurations

- **`results_analysis/`**: Analysis and visualization scripts
  - Performance metric computation
  - Plotting utilities for experimental results
  - Voltage distribution analysis

- **`runners/`**: Batch execution scripts
  - `sb3_runner.py`: Stable-Baselines3 integration
  - `run_exps.py`: Experiment orchestration

### Core Files

- **`train.py`**: Main training script
  - Supports multiple algorithms (TD3, PI-TD3, SAC, PI-SAC, PPO, etc.)
  - Configurable scenarios (V2G profit maximization, grid stability)
  - WandB integration for experiment tracking
  - Evaluation and model checkpointing

- **`example.py`**: Example usage and validation script
  - Demonstrates environment setup
  - Tests transition and reward function accuracy
  - Compares predicted vs. actual states

- **`batch_runner.py`**: SLURM batch job submission
  - Automated hyperparameter sweeps
  - Multi-seed experimental runs

- **`evaluator.py`**: Policy evaluation utilities
- **`generalization.py`**: Cross-scenario generalization testing

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- [Gurobi](https://www.gurobi.com/) (for optimal baseline comparisons)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/StavrosOrf/EV2Gym_PI-TD3.git
cd EV2Gym_PI-TD3
```

2. Create a virtual environment:
```bash
conda create -n ev2gym_pi python=3.8
conda activate ev2gym_pi
```

3. Install dependencies:
```bash
pip install torch gymnasium numpy pyyaml wandb tqdm pandas matplotlib
pip install -e ev2gym/  # Install EV2Gym environment
```

4. (Optional) Install Gurobi for optimal baselines:
```bash
# Follow instructions at https://www.gurobi.com/
```

## Execution Instructions

### Training a PI-TD3 Agent

Basic training command:
```bash
python train.py --policy pi_td3 --scenario grid_v2g_profitmax --K 5 --seed 0
```

### Command-line Arguments

**Algorithm Selection:**
- `--policy`: Algorithm to use (`pi_td3`, `td3`, `pi_sac`, `sac`, `ppo`, `shac`, `sapo`)
- `--K`: Lookahead horizon for physics-informed methods (default: 2)

**Scenario Configuration:**
- `--scenario`: Training scenario
  - `v2g_profitmax`: Basic V2G profit maximization
  - `grid_v2g_profitmax`: V2G with grid voltage constraints
  - `pst_v2g_profitmax`: Power system test scenario
- `--config`: Path to YAML config file (auto-selected based on scenario if not provided)

**Training Parameters:**
- `--seed`: Random seed (default: 0)
- `--device`: Device to use (`cuda:0`, `cpu`)
- `--max_timesteps`: Maximum training timesteps (default: 1e6)
- `--batch_size`: Training batch size (default: 256)
- `--discount`: Discount factor γ (default: 0.99)
- `--lambda_`: TD(λ) parameter (default: 0.95)

**Logging:**
- `--group_name`: WandB experiment group name
- `--disable_development_mode`: Disable dev mode (enables full WandB logging)
- `--lightweight_wandb`: Log fewer metrics

**Example Commands:**

Train PI-TD3 on grid scenario with 5-step lookahead:
```bash
python train.py \
  --policy pi_td3 \
  --scenario grid_v2g_profitmax \
  --K 5 \
  --seed 42 \
  --device cuda:0 \
  --group_name pi_td3_grid_experiments
```

Train standard TD3 baseline:
```bash
python train.py \
  --policy td3 \
  --scenario grid_v2g_profitmax \
  --seed 42 \
  --device cuda:0
```

Compare with different lookahead horizons:
```bash
for K in 1 2 5 10; do
  python train.py --policy pi_td3 --K $K --seed 0
done
```

### Testing and Evaluation

Run the example script to validate physics models:
```bash
python example.py
```

Evaluate a trained model:
```bash
python evaluator.py --model_path ./saved_models/your_model --config your_config.yaml
```

### Batch Experiments

For running multiple experiments (requires SLURM):
```bash
python batch_runner.py
```

Edit `batch_runner.py` to configure:
- Algorithms to compare
- Hyperparameter ranges
- Seeds and scenarios
- Computational resources

## Configuration Files

Configuration files in `config_files/` specify environment parameters:

- **Grid topology**: Number of buses, impedance matrices
- **EV parameters**: Battery capacity, charging rates, arrival/departure distributions
- **Economic parameters**: Electricity prices, user satisfaction weights
- **Simulation settings**: Time horizon, time step resolution

Example configuration structure:
```yaml
simulation:
  simulation_length: 96  # 15-min intervals for 24 hours
  timescale: 15  # minutes

grid:
  num_buses: 34
  base_power: 10000  # kVA

evs:
  battery_capacity: 70  # kWh
  max_charge_power: 22  # kW
  min_battery: 15  # kWh
```

## Physics Models

### State Transition Function (`agent/transition_fn.py`)

The differentiable state transition models:
- **SoC dynamics**: Battery state-of-charge updates based on power flow
- **Voltage propagation**: Grid voltage calculations using power flow equations
- **Smooth approximations**: Differentiable replacements for discrete operations (sign, min, max, clamp)

Key features:
- Smooth hyperbolic tangent approximations for step functions
- Differentiable min/max operations using soft approximations
- Gradient-friendly clamping for battery constraints

### Reward Function (`agent/loss_fn.py`)

The differentiable reward function captures:
- **Energy costs**: Charging/discharging costs based on electricity prices
- **User satisfaction**: Penalties for unmet charging demands at departure
- **Voltage violations**: Grid stability constraints (0.95-1.05 p.u.)
- **Power flow**: Grid losses and constraint satisfaction

## Results and Analysis

Analysis scripts in `results_analysis/` provide:
- `plots_all_avg.py`: Average performance across seeds
- `metric_table.py`: LaTeX tables for paper results
- `visualize_voltage.py`: Voltage profile visualization
- `wandb_query.py`: Download results from WandB

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2024,
  title={Physics-Informed Reinforcement Learning for Electric Vehicle Charging Management},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
