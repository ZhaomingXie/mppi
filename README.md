# MPPI Control for Unitree Go2 Robot

High-performance Model Predictive Path Integral (MPPI) controller for the Unitree Go2 quadruped robot using MuJoCo physics simulation.

## Performance

- **~280k FPS** rollout performance (100 environments, horizon 20)
- **19x speedup** over step-by-step execution
- Highly optimized Python/MuJoCo implementation

## Features

- **Parallel Environment (`vec_env.py`)**: Batched multiprocessing with shared memory
- **Optimized MPPI (`mppi_control.py`)**: Pre-allocated buffers, early termination, fused operations
- **Fast Rollouts**: Freeze-on-done optimization, zero-copy data transfer
- **Efficient Physics**: Optimized kinematic references, minimal Python overhead

## Installation

```bash
# Install MuJoCo Python bindings
pip install mujoco

# Install additional dependencies
pip install numpy cloudpickle
```

## Usage

### Run MPPI Controller
```bash
mjpython mppi_control.py
```

### Run Benchmark
```bash
mjpython benchmark_mppi.py
```

### Test Single Environment
```bash
mjpython sim.py
```

## Architecture

- `env.py` - Go2 environment with PD control and gait generation
- `vec_env.py` - Parallel environment wrapper with shared memory
- `mppi_control.py` - MPPI controller implementation
- `benchmark_mppi.py` - Performance benchmarking

## Optimizations Applied

1. Environment batching (10 workers for 100 envs)
2. Shared memory for zero-copy transfer
3. Fused set_state + rollout operations
4. Freeze-on-done for failed trajectories
5. Pre-computed kinematic reference lookup
6. Pre-allocated noise buffers
7. Optimized observation computation
8. Direct indexing without temporary arrays

## Configuration

Edit parameters in `mppi_control.py`:
- `HORIZON`: Planning horizon (default: 20)
- `N_SAMPLES`: Number of trajectory samples (default: 50-100)
- `N_ITERATIONS`: Refinement iterations (default: 50)
- `NOISE_SIGMA`: Exploration noise (default: 0.2)
- `LAMBDA`: Temperature parameter (default: 0.1)
