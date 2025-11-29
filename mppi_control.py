import numpy as np
import time
from vec_env import ParallelGo2Env
from env import Go2Env

class MPPIPlanner:
    def __init__(self, env, horizon=20, n_samples=64, n_iterations=5, noise_sigma=0.3, lambda_=1.0, convergence_threshold=1e-3):
        self.env = env
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.noise_sigma = noise_sigma
        self.lambda_ = lambda_
        self.convergence_threshold = convergence_threshold
        
        # Initialize mean action sequence (horizon, action_dim)
        self.action_dim = 12
        self.U = np.zeros((self.horizon, self.action_dim))
        
        # Pre-allocate noise buffer for efficiency
        self.noise_buffer = np.empty((self.n_samples, self.horizon, self.action_dim))
        
        # Ensure env has enough workers
        assert env.n_envs == n_samples, f"Env must have {n_samples} workers, got {env.n_envs}"

    def plan(self, current_state):
        # Run multiple iterations to refine the action sequence
        for iteration in range(self.n_iterations):
            # 1. Sample noise into pre-allocated buffer
            # Note: np.random.randn fills the buffer more efficiently than normal()
            self.noise_buffer[:] = np.random.randn(self.n_samples, self.horizon, self.action_dim) * self.noise_sigma
            
            # 2. Apply noise to mean action sequence and clip
            # Broadcasting U (T, dim) to (K, T, dim)
            actions = np.clip(self.U + self.noise_buffer, -1.0, 1.0)
            
            # 3. Rollout (fused with set_state)
            rewards, dones = self.env.rollout(actions, state=current_state)
            
            # 4. Compute costs
            # Sum rewards over time (negative for cost) + done penalty
            costs = np.sum(-rewards + dones * 10.0, axis=1)
                
            # 5. Compute weights
            beta = np.min(costs)
            weights = np.exp(-1.0 / self.lambda_ * (costs - beta))
            
            # Normalize weights
            sum_weights = np.sum(weights)
            if sum_weights > 1e-10:
                weights /= sum_weights
            else:
                weights = np.ones(self.n_samples) / self.n_samples
                
            # 6. Update U
            # Use tensordot for efficient weighted sum
            weighted_noise = np.tensordot(weights, self.noise_buffer, axes=([0], [0]))
            
            # Check convergence before update
            update_norm = np.linalg.norm(weighted_noise)
            
            self.U += weighted_noise
            self.U = np.clip(self.U, -1.0, 1.0)
            
            # Early termination if converged
            if update_norm < self.convergence_threshold:
                break
        
        # 7. Return first action
        action = self.U[0].copy()
        
        # 8. Shift U for next timestep
        self.U[:-1] = self.U[1:]
        self.U[-1] = np.zeros(self.action_dim)
        
        return action

def main():
    # Parameters
    HORIZON = 20
    N_SAMPLES = 50 # Number of parallel environments
    N_ITERATIONS = 50  # Number of iterations to refine per control step
    NOISE_SIGMA = 0.2
    LAMBDA = 0.1
    
    print(f"Initializing {N_SAMPLES} parallel environments for MPPI...")
    # Planning env (no render)
    planning_env = ParallelGo2Env(n_envs=N_SAMPLES, render_first=False)
    
    print("Initializing simulation environment...")
    # Execution env (render)
    sim_env = Go2Env(render=True)
    
    planner = MPPIPlanner(
        planning_env, 
        horizon=HORIZON, 
        n_samples=N_SAMPLES,
        n_iterations=N_ITERATIONS,
        noise_sigma=NOISE_SIGMA, 
        lambda_=LAMBDA
    )
    
    obs = sim_env.reset()
    print("Starting MPPI Optimization Phase...")
    
    # Phase 1: Optimize trajectory for a fixed number of steps
    OPTIMIZE_STEPS = 200
    optimized_actions = []
    
    try:
        total_reward = 0.0
        for step in range(OPTIMIZE_STEPS):
            start_time = time.time()
            
            # Get current state from sim_env including gait parameters
            qpos = sim_env.data.qpos.copy()
            qvel = sim_env.data.qvel.copy()
            phase = sim_env.phase
            sim_step_counter = sim_env.sim_step_counter
            speed = sim_env.speed
            mode = sim_env.mode
            
            # Full state tuple
            full_state = (qpos, qvel, phase, sim_step_counter, speed, mode)
            
            # Plan
            action = planner.plan(full_state)
            optimized_actions.append(action.copy())
            
            # Execute
            obs, reward, done, info = sim_env.step(action)
            total_reward += reward
            
            if step % 10 == 0:
                print(f"Optimization step {step}/{OPTIMIZE_STEPS}, Reward: {total_reward / (step + 1):.3f}")
            
            if done:
                print(f"Robot fell at step {step}. Stopping optimization.")
                optimized_actions = optimized_actions[:step]  # Truncate
                break
                
        print(f"\nOptimization complete! Generated {len(optimized_actions)} actions.")
        print("Now replaying trajectory in loop...")
        
        # Phase 2: Replay the optimized trajectory in a loop
        while True:
            obs = sim_env.reset()
            
            for step, action in enumerate(optimized_actions):
                obs, reward, done, info = sim_env.step(action)
                
                if done:
                    print(f"Robot fell at replay step {step}. Restarting trajectory.")
                    break
                    
                # Real-time playback
                time.sleep(sim_env.control_dt)
            
            print("Trajectory completed. Restarting...")
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        planning_env.close()
        sim_env.close()

if __name__ == "__main__":
    main()
