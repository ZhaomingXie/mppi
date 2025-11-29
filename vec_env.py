import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import os
from env import Go2Env

def worker(remote, parent_remote, env_fn_wrapper, n_envs_local, start_idx, shm_names, shapes, dtypes):
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

class ParallelGo2Env:
    def __init__(self, n_envs, render_first=False, max_horizon=100):
        self.n_envs = n_envs
        self.max_horizon = max_horizon
        self.action_dim = 12
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
                return Go2Env(render=render)
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
                self.dtypes
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

if __name__ == "__main__":
    # Test the parallel environment
    n_envs = 10
    env = ParallelGo2Env(n_envs, render_first=False)
    
    obs = env.reset()
    print(f"Reset done. Obs shape: {obs.shape}")
    
    start_time = time.time()
    steps = 1000
    
    try:
        # Test rollout
        horizon = 20
        actions = np.zeros((n_envs, horizon, 12))
        rewards, dones = env.rollout(actions)
        print(f"Rollout done. Rewards shape: {rewards.shape}")
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
