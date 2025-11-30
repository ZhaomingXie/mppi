import numpy as np
import time
from vec_env import ParallelGo2Env
from env import Go2Env

class ILQRPlanner:
    def __init__(self, env, horizon=20, iterations=10, precompute=True):
        self.env = env
        self.horizon = horizon
        self.iterations = iterations
        self.precompute = precompute
        
        # Dimensions
        self.qpos_dim = 19
        self.qvel_dim = 18
        self.state_dim = self.qpos_dim + self.qvel_dim # 37
        self.action_dim = 12
        
        # Cost weights
        self.Q = np.diag(np.concatenate([
            [10, 10, 10],          # pos (x,y,z)
            [10, 10, 10, 10],      # quat
            [10.0] * 12,            # joint angles
            [1.0] * 6,             # base vel
            [0.1] * 12             # joint vel
        ]))
        self.R = np.eye(self.action_dim) * 0.1
        
        # Initial guess for controls
        self.us = np.zeros((self.horizon, self.action_dim))
        
        # Perturbation for finite differences
        self.epsilon = 1e-2 # Increased from 1e-4 for contact stability
        
        # Precomputed derivatives storage
        self.fx_pre = {}
        self.fu_pre = {}
        
        if self.precompute:
            print("Precomputing derivatives around reference trajectory...")
            self._precompute_derivatives()

    def get_kinematic_reference(self, phase, speed, mode, sim_step_counter):
        # Re-implementing simplified version here for cost computation
        ref = np.zeros(19)
        ref[0] = speed * sim_step_counter * 0.02 # x pos
        ref[1:7] = [0, 0.38, 1.0, 0.0, 0.0, 0.0] # y, z, quat
        
        # Default leg positions
        ref[7::3] = 0      # x
        ref[8::3] = 0.65   # y
        ref[9::3] = -1.0   # z
        
        # Gait parameters
        max_phase = 20
        half_phase = 10
        phase_to_rad = 2 * np.pi / max_phase
        
        # Gait logic
        # speed=0 means stepping in place
        speed_factor_y = 0.4 * speed
        lift_factor_z = 0.9 # Lift height
        
        if phase < half_phase:
            # First half: FR (7,8,9) and RL (16,17,18) lift
            sin_phase = np.sin(phase * phase_to_rad)
            
            # FR
            ref[8] = 0.65 - speed_factor_y * sin_phase           
            ref[9] = -1.0 - lift_factor_z * sin_phase
            
            # RL
            ref[17] = 0.65 - speed_factor_y * sin_phase
            ref[18] = -1.0 - lift_factor_z * sin_phase
            
        else:
            # Second half: FL (10,11,12) and RR (13,14,15) lift
            sin_phase = np.sin((phase - half_phase) * phase_to_rad)
            
            # FL
            ref[11] = 0.65 - speed_factor_y * sin_phase
            ref[12] = -1.0 - lift_factor_z * sin_phase
            
            # RR
            ref[14] = 0.65 - speed_factor_y * sin_phase
            ref[15] = -1.0 - lift_factor_z * sin_phase
            
        return ref
            
    def _precompute_derivatives(self):
        # Compute Jacobians for all phases (0-19) around reference
        # We assume speed=0, mode=0 (standing/trotting in place) for now
        # Or we can just compute for the default gait parameters
        
        phases = range(20)
        x_refs = []
        u_refs = [] # Zero control
        aux_refs = []
        
        for p in phases:
            # Get kinematic reference state
            # We need to construct full state (qpos, qvel)
            # get_kinematic_reference returns qpos (19,)
            # We assume qvel is 0 for reference (approximation)
            xref = self.get_kinematic_reference(p, 0.0, 0, 0)
            xref_full = np.concatenate([xref, np.zeros(18)])
            
            x_refs.append(xref_full)
            u_refs.append(np.zeros(12))
            aux_refs.append((p, 0, 0.0, 0)) # sim_step_counter=0, speed=0, mode=0
            
        # We need to compute f(x) and f(x+eps)
        # 1. Compute f(x+eps) using compute_jacobians
        # We pass the list of references as a "trajectory"
        # compute_jacobians handles batching
        next_states_perturbed = self.env.compute_jacobians(x_refs, u_refs, aux_refs, self.epsilon)
        
        # 2. Compute f(x) (nominal)
        # We need to run 20 envs with the reference states
        # We can use the first 20 envs
        states = []
        for i in range(20):
            qpos = x_refs[i][:19]
            qvel = x_refs[i][19:]
            aux = aux_refs[i]
            states.append((qpos, qvel, *aux))
            
        # Fill rest with dummy
        while len(states) < self.env.n_envs:
            states.append(states[0])
            
        self.env.set_batch_state(states)
        
        # Step with zero actions
        actions = np.zeros((self.env.n_envs, 12))
        next_states_nominal, _, _ = self.env.step_dynamics(actions)
        
        # Compute gradients
        for t, p in enumerate(phases):
            fx = np.zeros((self.state_dim, self.state_dim))
            fu = np.zeros((self.state_dim, self.action_dim))
            
            x_next_nom = next_states_nominal[t]
            
            # State gradients
            for i in range(self.state_dim):
                if t % 2 == 0:
                    env_idx = i
                else:
                    env_idx = 50 + i
                
                x_next_pert = next_states_perturbed[env_idx, t]
                fx[:, i] = (x_next_pert - x_next_nom) / self.epsilon
                
            # Action gradients
            offset_u = self.state_dim
            for i in range(self.action_dim):
                p_idx = offset_u + i
                if t % 2 == 0:
                    env_idx = p_idx
                else:
                    env_idx = 50 + p_idx
                    
                x_next_pert = next_states_perturbed[env_idx, t]
                fu[:, i] = (x_next_pert - x_next_nom) / self.epsilon
                
            # Clamp
            fx = np.clip(fx, -100.0, 100.0)
            fu = np.clip(fu, -100.0, 100.0)
            
            self.fx_pre[p] = fx
            self.fu_pre[p] = fu
            
        print("Precomputation complete.")

    def compute_gradients(self, x_traj, u_traj, aux_traj):
        # Compute derivatives of dynamics f and cost l along the trajectory
        # x_traj: (H+1, state_dim)
        # u_traj: (H, action_dim)
        # aux_traj: list of (phase, sim_step_counter, speed, mode)
        
        H = self.horizon
        fx = np.zeros((H, self.state_dim, self.state_dim))
        fu = np.zeros((H, self.state_dim, self.action_dim))
        lx = np.zeros((H, self.state_dim))
        lu = np.zeros((H, self.action_dim))
        lxx = np.zeros((H, self.state_dim, self.state_dim))
        luu = np.zeros((H, self.action_dim, self.action_dim))
        lux = np.zeros((H, self.action_dim, self.state_dim))
        
        if not self.precompute:
            # Batch compute next states for all perturbations and all time steps
            # next_states_batch: (n_envs, H, state_dim)
            next_states_batch = self.env.compute_jacobians(x_traj[:-1], u_traj, aux_traj, self.epsilon)
            
            if np.any(np.isnan(next_states_batch)) or np.any(np.isinf(next_states_batch)):
                print("WARNING: NaN or Inf in next_states_batch!")
                next_states_batch = np.nan_to_num(next_states_batch)
        
        for t in range(H):
            x = x_traj[t]
            u = u_traj[t]
            phase, step_counter, speed, mode = aux_traj[t]
            
            if self.precompute:
                # Use precomputed gradients
                # Map phase to 0-19
                p_idx = int(phase) % 20
                fx[t] = self.fx_pre[p_idx]
                fu[t] = self.fu_pre[p_idx]
            else:
                # Compute gradients via Forward Difference
                # f'(x) = (f(x+eps) - f(x)) / eps
                # f(x) is x_traj[t+1] (nominal next state)
                
                x_next_nominal = x_traj[t+1]
                
                # State gradients
                for i in range(self.state_dim):
                    # vec_env mapping:
                    # If t is even: 0-49 (p_idx = i)
                    # If t is odd: 50-99 (p_idx = i)
                    
                    if t % 2 == 0:
                        env_idx = i
                    else:
                        env_idx = 50 + i
                        
                    x_next_perturbed = next_states_batch[env_idx, t]
                    fx[t, :, i] = (x_next_perturbed - x_next_nominal) / self.epsilon
                    
                # Action gradients
                offset_u = self.state_dim # 37
                for i in range(self.action_dim):
                    p_idx = offset_u + i
                    
                    if t % 2 == 0:
                        env_idx = p_idx
                    else:
                        env_idx = 50 + p_idx
                        
                    x_next_perturbed = next_states_batch[env_idx, t]
                    fu[t, :, i] = (x_next_perturbed - x_next_nominal) / self.epsilon
                    
                # Clamp gradients to avoid explosion
                fx[t] = np.clip(fx[t], -100.0, 100.0)
                fu[t] = np.clip(fu[t], -100.0, 100.0)
            
            if t == 0 and not self.precompute:
                # Debug print for first step
                # print(f"Max fx: {np.max(np.abs(fx[t]))}, Max fu: {np.max(np.abs(fu[t]))}")
                pass
                
            # Cost derivatives (analytical)
            # J = (x - xref)^T Q (x - xref) + u^T R u
            xref = self.get_kinematic_reference(phase, speed, mode, step_counter)
            xref_full = np.concatenate([xref, np.zeros(18)]) # velocity ref is 0 for now
            
            dx = x - xref_full
            lx[t] = self.Q @ dx
            lxx[t] = self.Q
            
            lu[t] = self.R @ u
            luu[t] = self.R
            
        return fx, fu, lx, lu, lxx, luu, lux


    def backward_pass(self, fx, fu, lx, lu, lxx, luu, lux):
        H = self.horizon
        k = np.zeros((H, self.action_dim))
        K = np.zeros((H, self.action_dim, self.state_dim))
        
        Vx = lx[-1].copy() # Terminal cost derivative (approximate with last step cost)
        Vxx = np.eye(self.state_dim) # Terminal cost Hessian (Identity)
        
        for t in range(H - 1, -1, -1):
            Qx = lx[t] + fx[t].T @ Vx
            Qu = lu[t] + fu[t].T @ Vx
            Qxx = lxx[t] + fx[t].T @ Vxx @ fx[t]
            Quu = luu[t] + fu[t].T @ Vxx @ fu[t]
            Qux = lux[t] + fu[t].T @ Vxx @ fx[t]
            
            # Regularization
            Quu_evals, Quu_evecs = np.linalg.eigh(Quu)
            Quu_evals[Quu_evals < 0] = 0.0
            Quu_evals += 1e-3 # Increased regularization
            Quu_inv = Quu_evecs @ np.diag(1.0 / Quu_evals) @ Quu_evecs.T
            
            k[t] = -Quu_inv @ Qu
            K[t] = -Quu_inv @ Qux
            
            Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            
        return k, K

    def forward_pass(self, x0, us, k, K, aux_start):
        H = self.horizon
        xs = np.zeros((H + 1, self.state_dim))
        us_new = np.zeros((H, self.action_dim))
        xs[0] = x0
        
        phase, step_counter, speed, mode = aux_start
        
        current_x = x0
        current_aux = (phase, step_counter, speed, mode)
        
        for t in range(H):
            # Calculate control
            # u = u_bar + k + K(x - x_bar)
            # But here we are just doing u = u_old + k + K(x - x_old)
            # Let's stick to standard: u_new = u_nom + k + K(x_new - x_nom)
            # We need x_nom (trajectory from previous iteration)
            # For simplicity in this first pass, let's assume us is u_nom
            # and we need x_nom.
            # Actually, standard iLQR uses the deviations.
            pass
            
        # Implementing simple rollout for now
        # We need to run this sequentially or in parallel?
        # Forward pass is sequential.
        # But we can run line search in parallel if we want.
        pass

    def plan(self, initial_state):
        # initial_state: (qpos, qvel, phase, sim_step_counter, speed, mode)
        qpos, qvel, phase, sim_step_counter, speed, mode = initial_state
        x0 = np.concatenate([qpos, qvel])
        aux_start = (phase, sim_step_counter, speed, mode)
        
        # Shift previous solution for warm start
        self.us[:-1] = self.us[1:]
        self.us[-1] = self.us[-2] # Duplicate last action
        
        # Initial rollout
        xs = np.zeros((self.horizon + 1, self.state_dim))
        xs[0] = x0
        aux_traj = []
        
        curr_x = x0
        curr_aux = aux_start
        
        # 0. Initial trajectory
        # We can do this in parallel if we had multiple guesses, but here just one.
        # We need to step the environment to get the trajectory.
        # We can use the first env of the parallel envs.
        
        # For efficiency, let's just use the parallel env with 1 active env
        for t in range(self.horizon):
            aux_traj.append(curr_aux)
            self.env.set_batch_state([curr_aux + tuple([0,0])]*self.env.n_envs) # Hacky way to set state?
            # No, set_batch_state expects (qpos, qvel, phase, ...)
            # curr_aux is (phase, ...)
            # We need to construct full state tuple
            state_tuple = (curr_x[:19], curr_x[19:], *curr_aux)
            
            # Use just one env for rollout?
            # Actually, we can just use the parallel env and ignore others
            states = [state_tuple] * self.env.n_envs
            self.env.set_batch_state(states)
            next_states, _, _ = self.env.step_dynamics(np.array([self.us[t]] * self.env.n_envs))
            
            curr_x = next_states[0]
            xs[t+1] = curr_x
            
            # Update aux
            p, s, sp, m = curr_aux
            curr_aux = ((p + 1) % 20, s + 1, sp, m)
            
        # iLQR Loop
        for i in range(self.iterations):
            # 1. Compute Gradients
            fx, fu, lx, lu, lxx, luu, lux = self.compute_gradients(xs, self.us, aux_traj)
            
            # 2. Backward Pass
            k, K = self.backward_pass(fx, fu, lx, lu, lxx, luu, lux)
            
            # 3. Forward Pass (Parallel Line Search)
            alphas = [1.0, 0.5, 0.1]
            
            # Execute rollouts in parallel
            # rollout_feedback returns list of (x_traj, u_traj, alpha)
            results = self.env.rollout_feedback(x0, self.us, xs[:-1], k, K, alphas, aux_start)
            
            best_cost = float('inf')
            best_us = self.us
            best_xs = xs
            
            for res in results:
                new_xs, new_us, alpha = res
                
                # Compute cost
                cost = 0
                curr_aux_ls = aux_start
                
                for t in range(self.horizon):
                    # Simplified cost calculation
                    xref = self.get_kinematic_reference(*curr_aux_ls)
                    xref_full = np.concatenate([xref, np.zeros(18)])
                    dx_cost = new_xs[t] - xref_full
                    u = new_us[t]
                    cost += dx_cost.T @ self.Q @ dx_cost + u.T @ self.R @ u
                    
                    # Update aux
                    p, s, sp, m = curr_aux_ls
                    curr_aux_ls = ((p + 1) % 20, s + 1, sp, m)
                    
                if cost < best_cost:
                    best_cost = cost
                    best_us = new_us
                    best_xs = new_xs
                    
            # Update trajectory
            self.us = best_us
            xs = best_xs
            
        return self.us[0]

def main():
    # Parameters
    HORIZON = 20 # Reduced from 20 for stability
    ITERATIONS = 5
    
    print("Initializing 100 parallel environments for iLQR...")
    planning_env = ParallelGo2Env(n_envs=100, render_first=False)
    
    print("Initializing simulation environment...")
    sim_env = Go2Env(render=True)
    
    planner = ILQRPlanner(planning_env, horizon=HORIZON, iterations=ITERATIONS)
    
    obs = sim_env.reset()
    print("Starting iLQR Control...")
    
    try:
        while True:
            start_time = time.time()
            
            # Get current state
            qpos = sim_env.data.qpos.copy()
            qvel = sim_env.data.qvel.copy()
            phase = sim_env.phase
            sim_step_counter = sim_env.sim_step_counter
            speed = sim_env.speed
            mode = sim_env.mode
            
            full_state = (qpos, qvel, phase, sim_step_counter, speed, mode)
            
            # Plan
            action = planner.plan(full_state)
            
            # Execute
            obs, reward, done, info = sim_env.step(action)
            
            if done:
                print("Robot fell. Resetting.")
                sim_env.reset()
                planner.us[:] = 0 # Reset warm start
                
            # Timing
            dt = time.time() - start_time
            print(f"Step time: {dt*1000:.1f}ms")
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        planning_env.close()
        sim_env.close()

if __name__ == "__main__":
    main()
