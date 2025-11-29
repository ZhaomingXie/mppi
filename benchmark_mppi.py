import time
import numpy as np
from vec_env import ParallelGo2Env

def benchmark_step(env, n_samples, horizon, n_iterations):
    print(f"Benchmarking step-by-step execution...")
    start_time = time.time()
    
    # Dummy actions
    actions = np.zeros((n_samples, horizon, 12))
    
    for _ in range(n_iterations):
        # Reset not strictly needed for perf test, but good for consistency
        # env.reset() 
        
        for t in range(horizon):
            step_actions = actions[:, t, :]
            obs, rewards, dones, infos = env.step(step_actions)
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"Step-by-step: {duration:.4f}s for {n_iterations} iterations")
    return duration

def benchmark_rollout(env, n_samples, horizon, n_iterations):
    print(f"Benchmarking vectorized rollout...")
    start_time = time.time()
    
    # Dummy actions
    actions = np.zeros((n_samples, horizon, 12))
    
    for _ in range(n_iterations):
        rewards, dones = env.rollout(actions)
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"Rollout:      {duration:.4f}s for {n_iterations} iterations")
    return duration

def main():
    N_SAMPLES = 200
    HORIZON = 20
    N_ITERATIONS = 50 # Number of rollouts to measure
    
    print(f"Initializing {N_SAMPLES} parallel environments...")
    env = ParallelGo2Env(n_envs=N_SAMPLES, render_first=False)
    env.reset()
    
    print(f"\nConfiguration:")
    print(f"  Parallel Envs: {N_SAMPLES}")
    print(f"  Horizon:       {HORIZON}")
    print(f"  Iterations:    {N_ITERATIONS}")
    print(f"  Total Steps:   {N_SAMPLES * HORIZON * N_ITERATIONS}")
    print("-" * 40)
    
    # Warmup
    print("Warming up...")
    env.step(np.zeros((N_SAMPLES, 12)))
    env.rollout(np.zeros((N_SAMPLES, HORIZON, 12)))
    print("-" * 40)

    # Benchmark Step
    time_step = benchmark_step(env, N_SAMPLES, HORIZON, N_ITERATIONS)
    
    # Benchmark Rollout
    time_rollout = benchmark_rollout(env, N_SAMPLES, HORIZON, N_ITERATIONS)
    
    print("-" * 40)
    print(f"Results:")
    print(f"  Step-by-step FPS: {(N_SAMPLES * HORIZON * N_ITERATIONS) / time_step:.0f}")
    print(f"  Rollout FPS:      {(N_SAMPLES * HORIZON * N_ITERATIONS) / time_rollout:.0f}")
    print(f"  Speedup:          {time_step / time_rollout:.2f}x")
    
    env.close()

if __name__ == "__main__":
    main()
