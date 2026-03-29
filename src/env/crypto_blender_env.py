import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CryptoBlenderEnv(gym.Env):
    """
    Gymnasium environment where the RL agent proposes sequences of gates
    to build an advanced hash function.
    """
    def __init__(self, max_steps=64):
        super(CryptoBlenderEnv, self).__init__()
        
        # Define action and observation space
        # Actions: Choose operation (ADD, XOR, ROTR, CH, MAJ), 
        # Choose operands (registers to act upon)
        self.num_ops = 5
        self.num_registers = 8
        self.max_steps = max_steps
        
        # Action: [op_type, target_reg, src_reg1, src_reg2]
        self.action_space = spaces.MultiDiscrete([self.num_ops, self.num_registers, self.num_registers, self.num_registers])
        
        # Observation space: the current sequence of operations 
        self.observation_space = spaces.Box(low=0, high=self.num_registers, shape=(self.max_steps, 4), dtype=np.int32)
        
        self.current_step = 0
        self.architecture_graph = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.architecture_graph = []
        obs = np.zeros((self.max_steps, 4), dtype=np.int32)
        return obs, {}

    def step(self, action):
        self.architecture_graph.append(action)
        self.current_step += 1
        
        # Reward is evaluated outside, normally at the end of the episode via avalanche criterion
        reward = 0.0
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        obs = np.zeros((self.max_steps, 4), dtype=np.int32)
        for i, a in enumerate(self.architecture_graph):
            obs[i] = a
            
        return obs, reward, terminated, truncated, {}
