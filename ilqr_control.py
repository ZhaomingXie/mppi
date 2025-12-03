import numpy as np
import time
from vec_env import ParallelEnv
from env import Go2Env
from spot_env import SpotEnv

class ILQRPlanner:
    def __init__(self, env, horizon=20, iterations=10, precompute=True, mode=0):
        self.env = env
        self.horizon = horizon
        self.iterations = iterations
        self.precompute = precompute
        self.mode = mode # 0: Trot, 1: Pace
        
        # Dimensions
        self.qpos_dim = env.qpos_dim
        self.qvel_dim = env.qvel_dim
        self.state_dim = self.qpos_dim + self.qvel_dim
        self.action_dim = env.action_dim
        
        # Cost weights
        # Q matrix depends on dimensions
        # pos (3), quat (4), joints (n_joints), base_vel (6), joint_vel (n_joints)
        n_joints = self.action_dim
        
        q_diag = np.concatenate([
            [1, 1, 10],          # pos (x,y,z)
            [10, 10, 10, 10],      # quat
            [10.0] * n_joints,     # joint angles
            [1.0] * 6,            # base vel
            [0.1] * n_joints       # joint vel
        ])
        
        self.Q = np.diag(q_diag)
        self.R = np.eye(self.action_dim) * 0.5
        
        # Initial guess for controls
        self.us = np.zeros((self.horizon, self.action_dim))
        
        # Perturbation for finite differences
        self.epsilon = 1e-2 # Increased from 1e-4 for contact stability
        
        # Precomputed derivatives storage
        self.fx_pre = {}
        self.fu_pre = {}
        
        # Check if precomputation is possible given environment count
        # We need (state_dim + action_dim) envs per timestep * 2 (for 2 timesteps in parallel)
        envs_needed_per_timestep = self.state_dim + self.action_dim
        envs_needed_total = envs_needed_per_timestep * 2
        
        if self.precompute and envs_needed_total > env.n_envs:
            print(f"WARNING: Precomputation requires {envs_needed_total} envs but only {env.n_envs} available.")
            print(f"Disabling precomputation. Using online gradient computation instead.")
            self.precompute = False
        
        if self.precompute:
            gait_name = "Trot" if self.mode == 0 else "Pace"
            print(f"Precomputing derivatives around {gait_name} reference trajectory...")
            self._precompute_derivatives()

            
    def get_kinematic_reference(self, phase, speed, mode, sim_step_counter):
        # Re-implementing simplified version here for cost computation
        ref_pos = np.zeros(self.qpos_dim)
        ref_vel = np.zeros(self.qvel_dim)
        
        # x pos
        ref_pos[0] = speed * sim_step_counter * 0.02 
        ref_vel[0] = speed # x vel
        
        # Base orientation/height
        if self.action_dim == 19: # Spot
            ref_pos[1:7] = [0, 0.6, 1.0, 0.0, 0.0, 0.0] # y, z, quat
        else: # Go2
            ref_pos[1:7] = [0, 0.38, 1.0, 0.0, 0.0, 0.0] # y, z, quat
        
        # Default leg positions
        if self.action_dim == 19: # Spot
            # fl, fr, hl, hr
            # 0, 1.04, -1.8
            nominal_leg = np.array([0, 0.6, -1.1])
            ref_pos[7:10] = nominal_leg
            ref_pos[10:13] = nominal_leg
            ref_pos[13:16] = nominal_leg
            ref_pos[16:19] = nominal_leg
            
            # Arm (7 joints) - keep at home
            # 0 -3.14 3.06 0 0 0 0
            ref_pos[19:] = [0, -3.14, 3.06, 0, 0, 0, 0]
            
        else: # Go2
            ref_pos[7::3] = 0      # x
            ref_pos[8::3] = 0.65   # y
            ref_pos[9::3] = -1.0   # z
        
        # Gait parameters
        max_phase = 30 if self.action_dim == 19 else 20
        half_phase = 15 if self.action_dim == 19 else 10
        phase_to_rad = 2 * np.pi / max_phase
        dt = 0.02
        omega = phase_to_rad / dt
        
        # Gait logic
        # speed=0 means stepping in place
        speed_factor_y = 0.4 * speed
        lift_factor_z = 1.0 # Lift height
        
        if self.action_dim == 19: # Spot
             # Reduced factors for Spot initially
            speed_factor_y = 0.2 * speed
            lift_factor_z = 0.5
        
        sin_phase = np.sin(phase * phase_to_rad)
        cos_phase = np.cos(phase * phase_to_rad)
        
        vy = -speed_factor_y * omega * cos_phase
        vz = -lift_factor_z * omega * cos_phase
        
        if phase < half_phase:
            # First half
            if mode == 0: # Trot
                if self.action_dim == 19: # Spot (FR=10.., HL=13..)
                    # FR (10,11,12) -> vel (9,10,11)
                    # Note: Spot leg indices: 0=hx, 1=hy, 2=kn.
                    # Go2: 0=hx, 1=hy, 2=kn.
                    # Logic is same, just indices.
                    
                    # FR (10,11,12)
                    ref_pos[11] = nominal_leg[1] + speed_factor_y * sin_phase
                    ref_pos[12] = nominal_leg[2] - lift_factor_z * sin_phase
                    ref_vel[10] = vy
                    ref_vel[11] = vz
                    
                    # HL (13,14,15) -> vel (12,13,14)
                    ref_pos[14] = nominal_leg[1] + speed_factor_y * sin_phase
                    ref_pos[15] = nominal_leg[2] - lift_factor_z * sin_phase
                    ref_vel[13] = vy
                    ref_vel[14] = vz
                    
                else: # Go2 (FR=7.., RL=16..)
                    # FR (7,8,9) -> vel (6,7,8)
                    ref_pos[8] = 0.65 - speed_factor_y * sin_phase           
                    ref_pos[9] = -1.0 - lift_factor_z * sin_phase
                    ref_vel[7] = vy
                    ref_vel[8] = vz
                    
                    # RL (16,17,18) -> vel (15,16,17)
                    ref_pos[17] = 0.65 - speed_factor_y * sin_phase
                    ref_pos[18] = -1.0 - lift_factor_z * sin_phase
                    ref_vel[16] = vy
                    ref_vel[17] = vz
                
            else: # Pace
                if self.action_dim == 19: # Spot (FR=10.., HR=16..)
                    # FR
                    ref_pos[11] = nominal_leg[1] + speed_factor_y * sin_phase
                    ref_pos[12] = nominal_leg[2] - lift_factor_z * sin_phase
                    ref_vel[10] = vy
                    ref_vel[11] = vz
                    
                    # HR
                    ref_pos[17] = nominal_leg[1] + speed_factor_y * sin_phase
                    ref_pos[18] = nominal_leg[2] - lift_factor_z * sin_phase
                    ref_vel[16] = vy
                    ref_vel[17] = vz
                else: # Go2 (FR=7.., RR=13..)
                    # FR
                    ref_pos[8] = 0.65 - speed_factor_y * sin_phase           
                    ref_pos[9] = -1.0 - lift_factor_z * sin_phase
                    ref_vel[7] = vy
                    ref_vel[8] = vz
                    
                    # RR
                    ref_pos[14] = 0.65 - speed_factor_y * sin_phase
                    ref_pos[15] = -1.0 - lift_factor_z * sin_phase
                    ref_vel[13] = vy
                    ref_vel[14] = vz
            
        else:
            # Second half
            sin_phase = np.sin((phase - half_phase) * phase_to_rad)
            cos_phase = np.cos((phase - half_phase) * phase_to_rad)
            
            vy = -speed_factor_y * omega * cos_phase
            vz = -lift_factor_z * omega * cos_phase
            
            if mode == 0: # Trot
                if self.action_dim == 19: # Spot (FL=7.., HR=16..)
                    # FL
                    ref_pos[8] = nominal_leg[1] + speed_factor_y * sin_phase
                    ref_pos[9] = nominal_leg[2] - lift_factor_z * sin_phase
                    ref_vel[7] = vy
                    ref_vel[8] = vz
                    
                    # HR
                    ref_pos[17] = nominal_leg[1] + speed_factor_y * sin_phase
                    ref_pos[18] = nominal_leg[2] - lift_factor_z * sin_phase
                    ref_vel[16] = vy
                    ref_vel[17] = vz
                else: # Go2 (FL=10.., RR=13..)
                    # FL
                    ref_pos[11] = 0.65 - speed_factor_y * sin_phase
                    ref_pos[12] = -1.0 - lift_factor_z * sin_phase
                    ref_vel[10] = vy
                    ref_vel[11] = vz
                    
                    # RR
                    ref_pos[14] = 0.65 - speed_factor_y * sin_phase
                    ref_pos[15] = -1.0 - lift_factor_z * sin_phase
                    ref_vel[13] = vy
                    ref_vel[14] = vz
                    
            else: # Pace
                if self.action_dim == 19: # Spot (FL=7.., HL=13..)
                    # FL
                    ref_pos[8] = nominal_leg[1] + speed_factor_y * sin_phase
                    ref_pos[9] = nominal_leg[2] - lift_factor_z * sin_phase
                    ref_vel[7] = vy
                    ref_vel[8] = vz
                    
                    # HL
                    ref_pos[14] = nominal_leg[1] + speed_factor_y * sin_phase
                    ref_pos[15] = nominal_leg[2] - lift_factor_z * sin_phase
                    ref_vel[13] = vy
                    ref_vel[14] = vz
                else: # Go2 (FL=10.., RL=16..)
                    # FL
                    ref_pos[11] = 0.65 - speed_factor_y * sin_phase
                    ref_pos[12] = -1.0 - lift_factor_z * sin_phase
                    ref_vel[10] = vy
                    ref_vel[11] = vz
                    
                    # RL
                    ref_pos[17] = 0.65 - speed_factor_y * sin_phase
                    ref_pos[18] = -1.0 - lift_factor_z * sin_phase
                    ref_vel[16] = vy
                    ref_vel[17] = vz
            
        return ref_pos, ref_vel
            
    def _precompute_derivatives(self):
        # Compute Jacobians for all phases (0-19) around reference
        # We assume speed=0, mode=self.mode (standing/trotting/pacing in place)
        
        phases = range(20) if self.action_dim == 12 else range(30)
        x_refs = []
        u_refs = [] # Zero control
        aux_refs = []
        
        for p in phases:
            # Get kinematic reference state
            xref_pos, xref_vel = self.get_kinematic_reference(p, 0.0, self.mode, 0)
            xref_full = np.concatenate([xref_pos, xref_vel])
            
            x_refs.append(xref_full)
            u_refs.append(np.zeros(self.action_dim))
            aux_refs.append((p, 0, 0.0, self.mode)) # sim_step_counter=0, speed=0, mode=self.mode
            
        # We need to compute f(x) and f(x+eps)
        # 1. Compute f(x+eps) using compute_jacobians
        # We pass the list of references as a "trajectory"
        # compute_jacobians handles batching
        next_states_perturbed = self.env.compute_jacobians(x_refs, u_refs, aux_refs, self.epsilon)
        
        # 2. Compute f(x) (nominal)
        # We need to run 20 envs with the reference states
        # We can use the first 20 envs
        states = []
        for i in range(len(phases)):
            qpos = x_refs[i][:self.qpos_dim]
            qvel = x_refs[i][self.qpos_dim:]
            aux = aux_refs[i]
            states.append((qpos, qvel, *aux))
            
        # Fill rest with dummy
        while len(states) < self.env.n_envs:
            states.append(states[0])
            
        self.env.set_batch_state(states)
        
        # Step with zero actions
        actions = np.zeros((self.env.n_envs, self.action_dim))
        next_states_nominal, _, _ = self.env.step_dynamics(actions)
        
        # Compute gradients
        for t, p in enumerate(phases):
            fx = np.zeros((self.state_dim, self.state_dim))
            fu = np.zeros((self.state_dim, self.action_dim))
            
            x_next_nom = next_states_nominal[t]
            
            half_envs = self.env.n_envs // 2
            
            # State gradients
            for i in range(self.state_dim):
                if t % 2 == 0:
                    env_idx = i
                else:
                    env_idx = half_envs + i
                
                x_next_pert = next_states_perturbed[env_idx, t]
                fx[:, i] = (x_next_pert - x_next_nom) / self.epsilon
                
            # Action gradients
            offset_u = self.state_dim
            for i in range(self.action_dim):
                p_idx = offset_u + i
                if t % 2 == 0:
                    env_idx = p_idx
                else:
                    env_idx = half_envs + p_idx
                    
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
                # Map phase to valid range
                max_phase = 30 if self.action_dim == 19 else 20
                p_idx = int(phase) % max_phase
                fx[t] = self.fx_pre[p_idx]
                fu[t] = self.fu_pre[p_idx]
            else:
                # Compute gradients via Forward Difference
                # f'(x) = (f(x+eps) - f(x)) / eps
                # f(x) is x_traj[t+1] (nominal next state)
                
                x_next_nominal = x_traj[t+1]
                
                # State gradients
                half_envs = self.env.n_envs // 2
                
                for i in range(self.state_dim):
                    # vec_env mapping:
                    # If t is even: 0 to half_envs-1 (p_idx = i)
                    # If t is odd: half_envs to n_envs-1 (p_idx = i)
                    
                    if t % 2 == 0:
                        env_idx = i
                    else:
                        env_idx = half_envs + i
                        
                    x_next_perturbed = next_states_batch[env_idx, t]
                    fx[t, :, i] = (x_next_perturbed - x_next_nominal) / self.epsilon
                    
                # Action gradients
                offset_u = self.state_dim # 37
                for i in range(self.action_dim):
                    p_idx = offset_u + i
                    
                    if t % 2 == 0:
                        env_idx = p_idx
                    else:
                        env_idx = half_envs + p_idx
                        
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
            xref_pos, xref_vel = self.get_kinematic_reference(phase, speed, mode, step_counter)
            xref_full = np.concatenate([xref_pos, xref_vel])
            
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
        Vxx = lxx[-1].copy() # Terminal cost Hessian (Use running cost Hessian Q instead of Identity)
        
        for t in range(H - 1, -1, -1):
            Qx = lx[t] + fx[t].T @ Vx
            Qu = lu[t] + fu[t].T @ Vx
            Qxx = lxx[t] + fx[t].T @ Vxx @ fx[t]
            Quu = luu[t] + fu[t].T @ Vxx @ fu[t]
            Qux = lux[t] + fu[t].T @ Vxx @ fx[t]
            
            # Regularization
            Quu_evals, Quu_evecs = np.linalg.eigh(Quu)
            Quu_evals[Quu_evals < 0] = 0.0
            Quu_evals += 1e-2 # Increased regularization for stability
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

    def compute_trajectory_cost(self, xs, us, aux_start):
        H = self.horizon
        
        # Pre-allocate reference array
        # Pre-allocate reference array
        x_refs = np.zeros((H, self.state_dim))
        
        p, s, sp, m = aux_start
        
        # Generate references
        for t in range(H):
            ref_pos, ref_vel = self.get_kinematic_reference(p, sp, m, s)
            x_refs[t, :self.qpos_dim] = ref_pos
            x_refs[t, self.qpos_dim:] = ref_vel
            
            max_phase = 30 if self.action_dim == 19 else 20
            p = (p + 1) % max_phase
            s += 1
            
        # Vectorized cost
        dx = xs[:H] - x_refs
        
        # State cost (Q is diagonal)
        Q_diag = np.diag(self.Q)
        state_cost = np.sum((dx**2) * Q_diag)
        
        # Control cost (R is diagonal)
        R_diag = np.diag(self.R)
        control_cost = np.sum((us**2) * R_diag)
        
        return state_cost + control_cost

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
        # Optimize: Use rollout_feedback with zero gains to simulate just 1 env
        # instead of stepping all 100 envs in a loop
        
        # Dummy gains
        k_dummy = np.zeros((self.horizon, self.action_dim))
        K_dummy = np.zeros((self.horizon, self.action_dim, self.state_dim))
        
        # Dummy x_nom (not used since K=0)
        x_nom_dummy = np.zeros((self.horizon, self.state_dim))
        
        # Single alpha to trigger 1 env
        alphas_dummy = [0.0]
        
        results = self.env.rollout_feedback(x0, self.us, x_nom_dummy, k_dummy, K_dummy, alphas_dummy, aux_start)
        
        # Extract result
        if results and results[0] is not None:
            xs_new, _, _ = results[0]
            xs = xs_new
            
            # Reconstruct aux_traj for gradient computation
            # rollout_feedback doesn't return aux_traj, so we regenerate it locally
            # This is fast (just integer math)
            curr_aux = aux_start
            max_phase = 30 if self.action_dim == 19 else 20
            for t in range(self.horizon):
                aux_traj.append(curr_aux)
                p, s, sp, m = curr_aux
                curr_aux = ((p + 1) % max_phase, s + 1, sp, m)
        else:
            # Fallback (should not happen)
            print("WARNING: Initial rollout failed, falling back to loop")
            curr_x = x0
            curr_aux = aux_start
            for t in range(self.horizon):
                aux_traj.append(curr_aux)
                state_tuple = (curr_x[:self.qpos_dim], curr_x[self.qpos_dim:], *curr_aux)
                states = [state_tuple] * self.env.n_envs
                self.env.set_batch_state(states)
                next_states, _, _ = self.env.step_dynamics(np.array([self.us[t]] * self.env.n_envs))
                curr_x = next_states[0]
                xs[t+1] = curr_x
                p, s, sp, m = curr_aux
                max_phase = 30 if self.action_dim == 19 else 20
                curr_aux = ((p + 1) % max_phase, s + 1, sp, m)
            
        # iLQR Loop
        for i in range(self.iterations):
            # 1. Compute Gradients
            fx, fu, lx, lu, lxx, luu, lux = self.compute_gradients(xs, self.us, aux_traj)
            
            # 2. Backward Pass
            k, K = self.backward_pass(fx, fu, lx, lu, lxx, luu, lux)
            
            # 3. Forward Pass (Parallel Line Search)
            alphas = [1.0, 0.5, 0.1]
            
            # Execute rollouts in parallel
            results = self.env.rollout_feedback(x0, self.us, xs[:-1], k, K, alphas, aux_start)
            
            best_cost = float('inf')
            best_us = self.us
            best_xs = xs
            
            for res in results:
                if res is None: continue
                new_xs, new_us, alpha = res
                
                # Vectorized cost computation
                cost = self.compute_trajectory_cost(new_xs, new_us, aux_start)
                    
                if cost < best_cost:
                    best_cost = cost
                    best_us = new_us
                    best_xs = new_xs
                    
            # Update trajectory
            self.us = best_us
            xs = best_xs
            
        # Recompute K around the final trajectory to get the correct feedback gain
        # This is crucial because the K from the loop was for the PREVIOUS trajectory
        fx, fu, lx, lu, lxx, luu, lux = self.compute_gradients(xs, self.us, aux_traj)
        k, K = self.backward_pass(fx, fu, lx, lu, lxx, luu, lux)
            
        return self.us[0], K[0]

def main():
    # Parameters
    # Parameters
    HORIZON = 20 # Reduced from 20 for stability
    ITERATIONS = 10
    GAIT_MODE = 1 # 0: Trot, 1: Pace
    
    # Replay mode
    REPLAY_MODE = True
    NUM_PLAN_STEPS = 1000  # Number of steps to plan before replaying
    PERTURBATION_SCALE = 0.1 # Noise scale for testing robustness
    
    # Robot selection
    ROBOT = "spot" # "go2" or "spot"
    
    # Determine n_envs based on robot (Spot needs more due to larger state/action dims)
    n_envs = 200 if ROBOT == "spot" else 100
    
    print(f"Initializing {n_envs} parallel environments for iLQR ({ROBOT})...")
    
    if ROBOT == "go2":
        env_class = Go2Env
    else:
        env_class = SpotEnv
        
    planning_env = ParallelEnv(n_envs=n_envs, env_class=env_class, render_first=False)
    
    print("Initializing simulation environment...")
    sim_env = env_class(render=True)
    sim_env.mode = GAIT_MODE # Set gait mode
    
    planner = ILQRPlanner(planning_env, horizon=HORIZON, iterations=ITERATIONS, mode=GAIT_MODE)
    
    obs = sim_env.reset()
    print(f"Starting iLQR Control with {'Trot' if GAIT_MODE == 0 else 'Pace'} gait...")
    
    # Collections for replay
    recorded_states = []
    recorded_actions = []
    recorded_Ks = []
    
    try:
        step_count = 0
        planning = True
        
        while True:
            if planning and step_count < NUM_PLAN_STEPS:
                # PLANNING MODE
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
                action, K = planner.plan(full_state)
                
                # Record
                if REPLAY_MODE:
                    recorded_states.append(full_state)
                    recorded_actions.append(action.copy())
                    recorded_Ks.append(K.copy())
                
                # Execute
                obs, reward, done, info = sim_env.step(action + np.random.normal(scale=PERTURBATION_SCALE, size=action.shape))
                
                if done:
                    print("Robot fell. Resetting.")
                    sim_env.reset()
                    planner.us[:] = 0 # Reset warm start
                    if REPLAY_MODE:
                        # Clear recordings and restart
                        recorded_states.clear()
                        recorded_actions.clear()
                        recorded_Ks.clear()
                        step_count = 0
                        continue
                    
                # Timing
                dt = time.time() - start_time
                print(f"Planning step {step_count+1}/{NUM_PLAN_STEPS}: {dt*1000:.1f}ms")
                
                step_count += 1
                
                if step_count >= NUM_PLAN_STEPS and REPLAY_MODE:
                    planning = False
                    print(f"\n{'='*50}")
                    print(f"SWITCHING TO REPLAY MODE (WITH FEEDBACK & PERTURBATION)")
                    print(f"{'='*50}")
                    print(f"Replaying {len(recorded_actions)} steps at real-time speed...")
                    print(f"Applying random action noise (scale={PERTURBATION_SCALE})...")
                    sim_env.reset()
                    
            else:
                # REPLAY MODE
                if not REPLAY_MODE or len(recorded_actions) == 0:
                    print("No replay enabled or no data recorded. Exiting.")
                    break
                
                # Loop replay continuously
                while True:
                    for i, (state, action) in enumerate(zip(recorded_states, recorded_actions)):
                        start_time = time.time()
                        
                        # Get recorded nominal state (x_nom)
                        qpos_nom, qvel_nom, _, _, _, _ = state
                        x_nom = np.concatenate([qpos_nom, qvel_nom])
                        
                        # Get current state (x)
                        qpos_curr = sim_env.data.qpos.copy()
                        qvel_curr = sim_env.data.qvel.copy()
                        x_curr = np.concatenate([qpos_curr, qvel_curr])
                        
                        # Compute feedback action
                        # u = u_nom + K @ (x - x_nom)
                        K = recorded_Ks[i]
                        u_fb = action + K @ (x_curr - x_nom)
                        
                        # Add perturbation
                        noise = np.random.randn(planner.action_dim) * PERTURBATION_SCALE
                        u_fb += noise
                        
                        u_fb = np.clip(u_fb, -1.0, 1.0) # Clip to valid range
                        
                        # Execute action
                        obs, reward, done, info = sim_env.step(u_fb)
                        
                        # Real-time timing (50Hz = 0.02s)
                        elapsed = time.time() - start_time
                        sleep_time = sim_env.control_dt - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        
                        if (i + 1) % 20 == 0:
                            print(f"Replay: {i+1}/{len(recorded_actions)} steps")
                            
                        # If we drift too far or fall, maybe reset?
                        # But for now let's just let it run to test robustness
                    
                    print("\nReplay loop complete, restarting...")
                    sim_env.reset() # Reset to start state for next loop

                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        planning_env.close()
        sim_env.close()

if __name__ == "__main__":
    main()
