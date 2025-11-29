import time
import mujoco
import mujoco.viewer
import os
import numpy as np
import math

class Go2Env:
    def __init__(self, render=True):
        # Load model
        self.model_path = os.path.join(os.path.dirname(__file__), "unitree_mujoco/unitree_robots/go2/scene.xml")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.control_dt = 0.02 # 50Hz control
        self.sim_steps_per_control = int(self.control_dt / self.dt)
        
        # Robot state
        self.n_joints = 12
        self.qpos_dim = 19 # 7 base + 12 joints
        self.qvel_dim = 18 # 6 base + 12 joints
        
        # PD Gains (from C++)
        self.Kp = np.array([10.0] * 12)
        self.Kd = np.array([0.1] * 12)
        
        # Nominal configuration (from C++)
        # gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.0, 0.65, -1, 0.0, 0.65, -1, 0.0, 0.65, -1, 0.0, 0.65, -1;
        self.qpos_init = np.array([0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 
                                   0.0, 0.65, -1.0, 
                                   0.0, 0.65, -1.0, 
                                   0.0, 0.65, -1.0, 
                                   0.0, 0.65, -1.0])
        
        # Gait parameters
        self.phase = 0
        self.max_phase = 20
        self.sim_step_counter = 0
        self.max_sim_step = 1000 # Increased for python testing
        self.speed = 0.0
        self.mode = 0
        
        self.reference = np.zeros(19)
        self.action_mean = self.qpos_init[7:]
        self.action_std = 0.3
        
        # Pre-compute constants for performance
        self._two_pi = 2 * np.pi
        self._phase_to_rad = self._two_pi / self.max_phase
        self._half_max_phase = self.max_phase / 2
        
        # Pre-allocate arrays for get_observation to avoid repeated concatenation
        self._obs_buffer = np.empty(41)  # Total observation size: 3+4+12+6+12+1+1+1+1
        
        # Cache for reference computation
        self._ref_base = np.array([0, 0, 0.38, 1.0, 0.0, 0.0, 0.0])
        
        # Reward tracking
        self.total_reward = 0.0
        
    def reset(self):
        self.sim_step_counter = 0
        self.total_reward = 0.0
        self.phase = 0#np.random.randint(0, self.max_phase)
        self.speed = 0#(np.random.randint(0, 11) - 5) * 0.1
        self.mode = 0#np.random.randint(0, 2)

        self.get_kinematic_reference()
        
        # Reset MuJoCo state
        self.data.qpos[:] = self.reference
        self.data.qvel[:] = 0.0
        self.data.qvel[0] = self.speed # Set initial forward velocity
        
        mujoco.mj_forward(self.model, self.data)
        
        return self.get_observation()

    def set_state(self, state):
        # State can be:
        # - (qpos, qvel, phase, sim_step_counter, speed, mode)
        # - (qpos, qvel) for backward compatibility
        # - qpos for backward compatibility
        if isinstance(state, tuple):
            if len(state) == 6:
                # Full state with gait parameters
                qpos, qvel, phase, sim_step_counter, speed, mode = state
                self.data.qpos[:] = qpos
                self.data.qvel[:] = qvel
                self.phase = phase
                self.sim_step_counter = sim_step_counter
                self.speed = speed
                self.mode = mode
            elif len(state) == 2:
                # Just physics state
                qpos, qvel = state
                self.data.qpos[:] = qpos
                self.data.qvel[:] = qvel
        else:
            # Backward compatibility: if just qpos
            self.data.qpos[:] = state
            self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        # Action scaling
        self.get_kinematic_reference()
        
        # Action is additive to the kinematic reference
        target_q = action + self.reference[7:]
        
        # PD Control Loop
        for _ in range(self.sim_steps_per_control):
            # Compute torques
            current_q = self.data.qpos[7:]
            current_dq = self.data.qvel[6:]
            
            tau = self.Kp * (target_q - current_q) + self.Kd * (0 - current_dq)
            self.data.ctrl[:] = tau
            
            mujoco.mj_step(self.model, self.data)
            
        self.phase = (self.phase + 1) % self.max_phase
        self.sim_step_counter += 1
        
        if self.viewer:
            self.viewer.sync()
            
        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.is_terminal_state()
        
        self.total_reward += reward
        
        return obs, reward, done, {}
    
    def step_rollout(self, action):
        """Optimized step for rollouts that skips observation computation"""
        # Action scaling
        self.get_kinematic_reference()
        
        # Action is additive to the kinematic reference
        target_q = action + self.reference[7:]
        
        # PD Control Loop
        for _ in range(self.sim_steps_per_control):
            # Compute torques
            current_q = self.data.qpos[7:]
            current_dq = self.data.qvel[6:]
            
            tau = self.Kp * (target_q - current_q) + self.Kd * (0 - current_dq)
            self.data.ctrl[:] = tau
            
            mujoco.mj_step(self.model, self.data)
            
        self.phase = (self.phase + 1) % self.max_phase
        self.sim_step_counter += 1
        
        # Skip observation - we don't need it in rollouts
        reward = self.compute_reward()
        done = self.is_terminal_state()
        
        return reward, done


    def get_kinematic_reference(self):
        # Optimized kinematic reference computation
        self.reference[0] = self.speed * self.sim_step_counter * self.control_dt
        self.reference[1:7] = self._ref_base[1:7]  # Use cached values
        
        # Set default leg positions (vectorized)
        self.reference[7::3] = 0      # All x positions
        self.reference[8::3] = 0.65   # All y positions  
        self.reference[9::3] = -1.0   # All z positions
            
        # Gait logic - optimized to avoid function calls
        sin_phase = np.sin(self.phase * self._phase_to_rad)
        speed_factor_y = 0.4 * self.speed
        
        if self.phase <= self._half_max_phase:
            # First half of gait
            if self.mode == 0:  # Trot: FR (7) and RL (16)
                self.reference[8] = 0.65 - speed_factor_y * sin_phase
                self.reference[9] = -1.0 - 0.7 * sin_phase
                self.reference[17] = 0.65 - speed_factor_y * sin_phase
                self.reference[18] = -1.0 - 0.7 * sin_phase
            else:  # Pace: FR (7) and RR (13)
                self.reference[8] = 0.65 - speed_factor_y * sin_phase
                self.reference[9] = -1.0 - 0.7 * sin_phase
                self.reference[14] = 0.65 - speed_factor_y * sin_phase
                self.reference[15] = -1.0 - 0.7 * sin_phase
        else:
            # Second half of gait
            sin_shifted = np.sin((self.phase - self._half_max_phase) * self._phase_to_rad)
            if self.mode == 0:  # Trot: FL (10) and RR (13)
                self.reference[11] = 0.65 - speed_factor_y * sin_shifted
                self.reference[12] = -1.0 - 0.7 * sin_shifted
                self.reference[14] = 0.65 - speed_factor_y * sin_shifted
                self.reference[15] = -1.0 - 0.7 * sin_shifted
            else:  # Pace: FL (10) and RL (16)
                self.reference[11] = 0.65 - speed_factor_y * sin_shifted
                self.reference[12] = -1.0 - 0.7 * sin_shifted
                self.reference[17] = 0.65 - speed_factor_y * sin_shifted
                self.reference[18] = -1.0 - 0.7 * sin_shifted

    def get_observation(self):
        # Optimized observation computation using pre-allocated buffer
        # Total size: 3 + 4 + 12 + 6 + 12 + 1 + 1 + 1 + 1 = 41
        # Fill buffer directly to avoid repeated concatenation
        self._obs_buffer[0:3] = self.data.qpos[0:3]    # Position (3)
        self._obs_buffer[3:7] = self.data.qpos[3:7]    # Orientation (quat) (4)
        self._obs_buffer[7:19] = self.data.qpos[7:]    # Joint angles (12)
        self._obs_buffer[19:25] = self.data.qvel[:6]   # Body lin/ang vel (6)
        self._obs_buffer[25:37] = self.data.qvel[6:]   # Joint vel (12)
        self._obs_buffer[37] = self.speed              # Speed (1)
        self._obs_buffer[38] = self.mode               # Mode (1)
        
        # Use numpy sin/cos for consistency and pre-computed constant
        phase_angle = self.phase * self._phase_to_rad
        self._obs_buffer[39] = np.sin(phase_angle)     # sin(phase) (1)
        self._obs_buffer[40] = np.cos(phase_angle)     # cos(phase) (1)
        
        return self._obs_buffer.copy()  # Return copy for safety

    def compute_reward(self):
        # Ported from C++ computeReward
        # float joint_reward = 0, position_reward = 0, orientation_reward = 0;
        
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        # Joint reward
        joint_diff = qpos[7:] - self.reference[7:]
        joint_reward = np.sum(joint_diff**2)
        
        # Position reward (tracking x, y, z)
        # Note: C++ tracks absolute position reference_[0] which increases over time.
        # Here we do the same.
        pos_diff = self.reference[:3] - qpos[:3]
        position_reward = np.sum(pos_diff**2)
        
        # Orientation reward
        # C++: 2 * (diff_quat_x^2 + diff_quat_y^2 + diff_quat_z^2) + 5 * (ang_vel^2)
        # Note: C++ reference orientation is 1,0,0,0 (w,x,y,z) -> indices 3,4,5,6
        # C++ code: gc_[4]-reference_[4] ... indices 4,5,6 are x,y,z parts of quat in Raisim?
        # MuJoCo quat is w,x,y,z.
        # Let's assume we want to stay upright (unit quat).
        ori_diff = qpos[4:7] - self.reference[4:7] # x,y,z parts
        orientation_reward = 2 * np.sum(ori_diff**2)
        orientation_reward += 5 * np.sum(qvel[3:6]**2) # Angular velocity penalty
        
        r_pos = np.exp(-position_reward)
        r_ori = np.exp(-orientation_reward)
        r_joint = np.exp(-2 * joint_reward)
        
        return 0.3 * r_pos + 0.3 * r_ori + 0.4 * r_joint

    def is_terminal_state(self):
        # if (gc_[2] < 0.3) return true;
        # if (std::abs(gc_[3]) < 0.9) return true; // qw < 0.9 means tilted
        
        qpos = self.data.qpos
        
        if qpos[2] < 0.25: # Body height too low
            return True
            
        if abs(qpos[3]) < 0.9: # Tilted too much (w component of quat)
            return True
            
        # Contact check omitted for simplicity in first pass
        return False

    def close(self):
        if self.viewer:
            self.viewer.close()

if __name__ == "__main__":
    env = Go2Env(render=True)
    obs = env.reset()
    print("Reset done. Initial Observation:", obs.shape)
    
    try:
        while True:
            # Zero action (PD will try to track reference)
            action = np.zeros(12) 
            obs, reward, done, info = env.step(action)
            print(reward, done)
            
            if done:
                print("Episode finished. Total Reward:", env.total_reward)
                env.reset()
                
            time.sleep(env.control_dt)
            
    except KeyboardInterrupt:
        env.close()
