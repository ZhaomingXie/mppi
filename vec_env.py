import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import os
from env import Go2Env
from spot_env import SpotEnv

def worker(remote, parent_remote, env_fn_wrapper, n_envs_local, start_idx, shm_names, shapes, dtypes, n_envs_total, dims):
    parent_remote.close()
    
    # Initialize multiple environments
    envs = [env_fn_wrapper.x() for _ in range(n_envs_local)]
    
    # Attach to shared memory
    shm_actions = shared_memory.SharedMemory(name=shm_names['actions'])
    shm_rewards = shared_memory.SharedMemory(name=shm_names['rewards'])
    shm_dones = shared_memory.SharedMemory(name=shm_names['dones'])
    
    # Create numpy arrays backed by shared memory
    actions_buf = np.ndarray(shapes['actions'], dtype=dtypes['actions'], buffer=shm_actions.buf)
    rewards_buf = np.ndarray(shapes['rewards'], dtype=dtypes['rewards'], buffer=shm_rewards.buf)
    dones_buf = np.ndarray(shapes['dones'], dtype=dtypes['dones'], buffer=shm_dones.buf)
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                # data is actions for this worker's envs: (n_envs_local, action_dim)
                actions = data
                obs_list = []
                rew_list = []
                done_list = []
                info_list = []
                
                for i, env in enumerate(envs):
                    ob, reward, done, info = env.step(actions[i])
                    if done:
                        ob = env.reset()
                    obs_list.append(ob)
                    rew_list.append(reward)
                    done_list.append(done)
                    info_list.append(info)
                    
                remote.send((np.stack(obs_list), np.stack(rew_list), np.stack(done_list), info_list))
                
            elif cmd == 'step_dynamics':
                # Step without reset and return full state (qpos, qvel)
                # data is actions
                actions = data
                next_states = []
                # rewards = []
                # dones = []
                
                for i, env in enumerate(envs):
                    # ob, reward, done, info = env.step(actions[i])
                    env.step_minimal(actions[i])
                    
                    # Get full state
                    qpos = env.data.qpos.copy()
                    qvel = env.data.qvel.copy()
                    next_states.append(np.concatenate([qpos, qvel]))
                    # rewards.append(reward)
                    # dones.append(done)
                    
                remote.send((np.stack(next_states), None, None))

            elif cmd == 'compute_jacobians':
                # data is (x_traj, u_traj, aux_traj, epsilon)
                x_traj, u_traj, aux_traj, epsilon = data
                H = len(x_traj)
                
                # Results: (n_envs_local, H, state_dim)
                # We will populate this sparsely but deterministically
                state_dim = dims['state_dim']
                qpos_dim = dims['qpos_dim']
                qvel_dim = dims['qvel_dim']
                action_dim = dims['action_dim']
                
                results = np.zeros((n_envs_local, H, state_dim))
                
                # Iterate in blocks of 2 time steps
                for t_block in range(0, H, 2):
                    
                    for i, env in enumerate(envs):
                        global_idx = start_idx + i
                        
                        # Map global_idx to (t_offset, p_idx)
                        # 0 to half_envs-1: t_block
                        # half_envs to n_envs-1: t_block + 1
                        half_envs = n_envs_total // 2
                        
                        if global_idx < half_envs:
                            t_offset = 0
                            p_idx = global_idx
                        else:
                            t_offset = 1
                            p_idx = global_idx - half_envs
                            
                        t = t_block + t_offset
                        
                        if t >= H:
                            continue
                            
                        # Nominal data for time t
                        x_nominal = x_traj[t]
                        u_nominal = u_traj[t]
                        aux = aux_traj[t]
                        
                        qpos = x_nominal[:qpos_dim].copy()
                        qvel = x_nominal[qpos_dim:].copy()
                        u = u_nominal.copy()
                        
                        # Apply perturbation (Forward Difference)
                        # Apply perturbation (Forward Difference)
                        # p_idx 0 to state_dim-1: state
                        # state_dim to state_dim+action_dim-1: action
                        if p_idx < state_dim:
                            # x + eps
                            dim = p_idx
                            if dim < qpos_dim:
                                qpos[dim] += epsilon
                            else:
                                qvel[dim - qpos_dim] += epsilon
                        elif p_idx < state_dim + action_dim:
                            # u + eps
                            dim = p_idx - state_dim
                            u[dim] += epsilon
                        else:
                            # Idle env (e.g. index 49, 99)
                            continue
                            
                        # Set state
                        env.set_state((qpos, qvel, *aux))
                        
                        # Step
                        # ob, reward, done, info = env.step(u)
                        env.step_minimal(u)
                        
                        # Get next state
                        qpos_next = env.data.qpos.copy()
                        qvel_next = env.data.qvel.copy()
                        results[i, t] = np.concatenate([qpos_next, qvel_next])
                        
                remote.send(results)

            elif cmd == 'rollout_feedback':
                # data is (x0, u_nom, x_nom, k, K, alphas, aux_start)
                x0, u_nom, x_nom, k, K, alphas, aux_start = data
                H = len(u_nom)
                
                # Extract dimensions
                state_dim = dims['state_dim']
                qpos_dim = dims['qpos_dim']
                qvel_dim = dims['qvel_dim']
                action_dim = dims['action_dim']
                
                # Each env handles one alpha
                # We want to distribute alphas across workers to run in parallel
                # Instead of using first N envs (which might be on same worker),
                # we stride them across the total environments.
                
                n_alphas = len(alphas)
                stride = max(1, n_envs_total // n_alphas)
                
                results = []
                
                for i, env in enumerate(envs):
                    global_idx = start_idx + i
                    
                    # Check if this env is assigned a task
                    # We assign task j to env (j * stride)
                    # So we check if global_idx is a multiple of stride
                    # and if the corresponding task index is valid
                    
                    task_idx = -1
                    if n_alphas > 0:
                        if global_idx % stride == 0:
                            idx = global_idx // stride
                            if idx < n_alphas:
                                task_idx = idx
                    
                    if task_idx >= 0:
                        alpha = alphas[task_idx]
                        
                        # Initialize
                        qpos = x0[:qpos_dim]
                        qvel = x0[qpos_dim:]
                        env.set_state((qpos, qvel, *aux_start))
                        
                        curr_x = x0.copy()
                        curr_aux = aux_start
                        
                        x_traj = np.zeros((H + 1, state_dim))
                        u_traj = np.zeros((H, action_dim))
                        x_traj[0] = curr_x
                        
                        valid = True
                        for t in range(H):
                            # Control law
                            dx = curr_x - x_nom[t]
                            u = u_nom[t] + alpha * k[t] + K[t] @ dx
                            u = np.clip(u, -1.0, 1.0)
                            u_traj[t] = u
                            
                            # Step
                            # ob, reward, done, info = env.step(u)
                            env.step_minimal(u)
                            
                            # Get next state
                            qpos_next = env.data.qpos.copy()
                            qvel_next = env.data.qvel.copy()
                            curr_x = np.concatenate([qpos_next, qvel_next])
                            x_traj[t+1] = curr_x
                            
                            # Update aux (approximate)
                            p, s, sp, m = curr_aux
                            curr_aux = ((p + 1) % env.max_phase, s + 1, sp, m)
                                
                        results.append((x_traj, u_traj, alpha))
                    else:
                        results.append(None)
                        
                remote.send(results)



                
            elif cmd == 'reset':
                obs_list = [env.reset() for env in envs]
                remote.send(np.stack(obs_list))
                
            elif cmd == 'close':
                remote.close()
                break
                
            elif cmd == 'get_attr':
                # Return list of attrs
                remote.send([getattr(env, data) for env in envs])
                
            elif cmd == 'set_attr':
                for env in envs:
                    setattr(env, data[0], data[1])
                    
            elif cmd == 'set_state':
                # data is state. Set same state for all envs? Or list of states?
                # For MPPI, we usually set same state for all samples.
                # Let's assume same state for now.
                for env in envs:
                    env.set_state(data)
                remote.send(True)  # Confirm state was set
                
            elif cmd == 'set_batch_state':
                # data is a list/array of states, one for each env
                states = data
                for i, env in enumerate(envs):
                    env.set_state(states[i])
                remote.send(True)

                
            elif cmd == 'rollout':
                # data is (horizon, state)
                horizon, state = data
                
                # If state is provided, set it for all envs
                if state is not None:
                    for env in envs:
                        env.set_state(state)
                
                # Read actions from shared memory
                # My slice: start_idx : start_idx + n_envs_local
                my_actions = actions_buf[start_idx : start_idx + n_envs_local, :horizon, :]
                
                # Track done status for local envs
                local_dones = np.zeros(n_envs_local, dtype=bool)

                # Rollout loop
                for t in range(horizon):
                    for i in range(n_envs_local):
                        if local_dones[i]:
                            # Already done, skip simulation
                            rewards_buf[start_idx + i, t] = 0.0
                            dones_buf[start_idx + i, t] = True
                            continue
                        
                        # Step environment (optimized for rollouts - skips observation)
                        reward, done = envs[i].step_rollout(my_actions[i, t])
                        
                        # Write results to shared memory
                        rewards_buf[start_idx + i, t] = reward
                        dones_buf[start_idx + i, t] = done
                        
                        if done:
                            local_dones[i] = True
                            # Do NOT reset. Let it stay done.
                            # env.reset()
                
                remote.send(True) # Signal done
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except KeyboardInterrupt:
        print('Worker KeyboardInterrupt')
    finally:
        shm_actions.close()
        shm_rewards.close()
        shm_dones.close()
        for env in envs:
            env.close()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to pickle
    :param x: (Any) the object you wish to wrap
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class ParallelEnv:
    def __init__(self, n_envs, env_class=Go2Env, render_first=False, max_horizon=100):
        self.n_envs = n_envs
        self.max_horizon = max_horizon
        
        # Determine dimensions from a dummy env
        dummy_env = env_class(render=False)
        self.qpos_dim = dummy_env.qpos_dim
        self.qvel_dim = dummy_env.qvel_dim
        self.action_dim = getattr(dummy_env, 'action_dim', 12) # Default to 12 if not set (Go2Env doesn't set it explicitly in init but uses 12)
        # Actually Go2Env doesn't set action_dim in init, it's implicit. 
        # Let's check env.py... it says self.n_joints = 12.
        # SpotEnv sets self.n_joints = 19.
        # Let's use n_joints as action_dim.
        if hasattr(dummy_env, 'n_joints'):
            self.action_dim = dummy_env.n_joints
            
        self.state_dim = self.qpos_dim + self.qvel_dim
        dummy_env.close()
        
        self.dims = {
            'qpos_dim': self.qpos_dim,
            'qvel_dim': self.qvel_dim,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim
        }
        
        self.closed = False
        
        # Determine number of workers
        # Use min(n_envs, cpu_count)
        n_cpus = mp.cpu_count()
        self.n_workers = min(n_envs, n_cpus)
        
        # Distribute envs to workers
        self.envs_per_worker = [n_envs // self.n_workers] * self.n_workers
        for i in range(n_envs % self.n_workers):
            self.envs_per_worker[i] += 1
            
        print(f"ParallelGo2Env: {n_envs} envs -> {self.n_workers} workers {self.envs_per_worker}")
        
        def make_env(render):
            def _thunk():
                return env_class(render=render)
            return _thunk

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_workers)])
        self.ps = []
        
        # Allocate Shared Memory
        self.shapes = {
            'actions': (n_envs, max_horizon, self.action_dim),
            'rewards': (n_envs, max_horizon),
            'dones':   (n_envs, max_horizon)
        }
        self.dtypes = {
            'actions': np.float64,
            'rewards': np.float64,
            'dones':   bool
        }
        
        # Calculate sizes
        size_actions = int(np.prod(self.shapes['actions']) * np.dtype(self.dtypes['actions']).itemsize)
        size_rewards = int(np.prod(self.shapes['rewards']) * np.dtype(self.dtypes['rewards']).itemsize)
        size_dones   = int(np.prod(self.shapes['dones'])   * np.dtype(self.dtypes['dones']).itemsize)
        
        self.shm_actions = shared_memory.SharedMemory(create=True, size=size_actions)
        self.shm_rewards = shared_memory.SharedMemory(create=True, size=size_rewards)
        self.shm_dones   = shared_memory.SharedMemory(create=True, size=size_dones)
        
        self.shm_names = {
            'actions': self.shm_actions.name,
            'rewards': self.shm_rewards.name,
            'dones':   self.shm_dones.name
        }
        
        # Create numpy arrays backed by shared memory (for main process)
        self.actions_buf = np.ndarray(self.shapes['actions'], dtype=self.dtypes['actions'], buffer=self.shm_actions.buf)
        self.rewards_buf = np.ndarray(self.shapes['rewards'], dtype=self.dtypes['rewards'], buffer=self.shm_rewards.buf)
        self.dones_buf   = np.ndarray(self.shapes['dones'],   dtype=self.dtypes['dones'],   buffer=self.shm_dones.buf)
        
        start_idx = 0
        for i in range(self.n_workers):
            # Only render the first environment if requested (and it's in the first worker)
            render = (i == 0) and render_first
            
            n_envs_local = self.envs_per_worker[i]
            
            p = mp.Process(target=worker, args=(
                self.work_remotes[i], 
                self.remotes[i], 
                CloudpickleWrapper(make_env(render)),
                n_envs_local,
                start_idx,
                self.shm_names,
                self.shapes,
                self.dtypes,
                n_envs,
                self.dims
            ))
            p.daemon = True
            p.start()
            self.ps.append(p)
            
            start_idx += n_envs_local
            
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        # actions: (n_envs, action_dim)
        start_idx = 0
        for i, remote in enumerate(self.remotes):
            n_envs_local = self.envs_per_worker[i]
            remote.send(('step', actions[start_idx : start_idx + n_envs_local]))
            start_idx += n_envs_local
            
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        
        # Flatten results
        # obs is tuple of (n_workers, n_envs_local, obs_dim) -> (n_envs, obs_dim)
        return np.concatenate(obs), np.concatenate(rews), np.concatenate(dones), sum(infos, [])

    def step_dynamics(self, actions):
        # actions: (n_envs, action_dim)
        start_idx = 0
        for i, remote in enumerate(self.remotes):
            n_envs_local = self.envs_per_worker[i]
            remote.send(('step_dynamics', actions[start_idx : start_idx + n_envs_local]))
            start_idx += n_envs_local
            
        results = [remote.recv() for remote in self.remotes]
        next_states, rews, dones = zip(*results)
        
        if rews[0] is None:
            return np.concatenate(next_states), None, None
        else:
            return np.concatenate(next_states), np.concatenate(rews), np.concatenate(dones)


    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.concatenate([remote.recv() for remote in self.remotes])

    def set_state(self, state):
        for remote in self.remotes:
            remote.send(('set_state', state))
        # Wait for confirmation from all workers
        for remote in self.remotes:
            remote.recv()

    def set_batch_state(self, states):
        # states: list of states, length must be n_envs
        assert len(states) == self.n_envs
        
        start_idx = 0
        for i, remote in enumerate(self.remotes):
            n_envs_local = self.envs_per_worker[i]
            local_states = states[start_idx : start_idx + n_envs_local]
            remote.send(('set_batch_state', local_states))
            start_idx += n_envs_local
            
        # Wait for confirmation
        for remote in self.remotes:
            remote.recv()


    def compute_jacobians(self, x_traj, u_traj, aux_traj, epsilon=1e-4):
        # x_traj: (H, 37)
        # u_traj: (H, 12)
        # aux_traj: list of H tuples
        
        for remote in self.remotes:
            remote.send(('compute_jacobians', (x_traj, u_traj, aux_traj, epsilon)))
            
        results = [remote.recv() for remote in self.remotes]
        # results is list of (n_envs_local, H, 37)
        # Concatenate along env dimension
        return np.concatenate(results, axis=0)

        return np.concatenate(results, axis=0)

    def rollout_feedback(self, x0, u_nom, x_nom, k, K, alphas, aux_start):
        # Broadcast to all workers
        # But only first few envs will do work
        for remote in self.remotes:
            remote.send(('rollout_feedback', (x0, u_nom, x_nom, k, K, alphas, aux_start)))
            
        results = [remote.recv() for remote in self.remotes]
        # Flatten
        flat_results = []
        for r in results:
            flat_results.extend(r)
            
        # Filter None
        return [r for r in flat_results if r is not None]

    def rollout(self, actions, state=None):
        # actions: (n_envs, horizon, action_dim)
        n_envs, horizon, action_dim = actions.shape
        
        assert n_envs == self.n_envs
        assert horizon <= self.max_horizon
        
        # 1. Write actions to shared memory
        self.actions_buf[:, :horizon, :] = actions
        
        # 2. Signal workers to start
        # If state provided, send it. Otherwise send None.
        # We overload the 'rollout' command data to be (horizon, state)
        for remote in self.remotes:
            remote.send(('rollout', (horizon, state)))
            
        # 3. Wait for workers to finish
        for remote in self.remotes:
            remote.recv()
            
        # 4. Read results from shared memory (return views for zero-copy)
        rewards = self.rewards_buf[:, :horizon]
        dones = self.dones_buf[:, :horizon]
        
        return rewards, dones

    def close(self):
        if self.closed:
            return
        if self.remotes:
            for remote in self.remotes:
                remote.send(('close', None))
            for p in self.ps:
                p.join()
                
        # Clean up shared memory
        try:
            self.shm_actions.close()
            self.shm_actions.unlink()
        except FileNotFoundError:
            pass

        try:
            self.shm_rewards.close()
            self.shm_rewards.unlink()
        except FileNotFoundError:
            pass

        try:
            self.shm_dones.close()
            self.shm_dones.unlink()
        except FileNotFoundError:
            pass
        
        self.closed = True

    def __del__(self):
        self.close()

# Alias for backward compatibility
ParallelGo2Env = ParallelEnv

if __name__ == "__main__":
    # Test the parallel environment
    n_envs = 10
    # Test with Go2
    print("Testing Go2...")
    env = ParallelEnv(n_envs, env_class=Go2Env, render_first=False)
    obs = env.reset()
    print(f"Go2 Reset done. Obs shape: {obs.shape}")
    env.close()
    
    # Test with Spot
    print("\nTesting Spot...")
    env = ParallelEnv(n_envs, env_class=SpotEnv, render_first=False)
    obs = env.reset()
    print(f"Spot Reset done. Obs shape: {obs.shape}")
    env.close()

