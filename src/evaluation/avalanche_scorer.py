import numpy as np
from src.env.bitwise_ops import rotr, rotl, ch, maj, sigma0_256, sigma1_256, lower_sigma0_256, lower_sigma1_256, add_mod32

def hamming_distance(val1, val2):
    """Calculate the Hamming distance between two 32-bit integers."""
    return bin(val1 ^ val2).count('1')

def evaluate_avalanche_criterion(architecture_graph, num_samples=1000):
    """
    Evaluates the Strict Avalanche Criterion (SAC) of a proposed architecture.
    A good hash function should flip exactly 50% of its bits when 1 input bit is flopped.
    
    Returns a reward where higher is better (closer to 128 bit flips for a 256-bit hash).
    """
    total_penalty = 0.0
    
    # In a real scenario, this would apply the `architecture_graph` 
    # to generated message schedules and states `num_samples` times,
    # measuring the bit flip variance.
    for _ in range(num_samples):
        # Dummy simulation of bit flips (would be replaced by actual architecture execution)
        # We want bits_flipped to be exactly 128 on average for a 256-bit block.
        simulated_flips = np.random.normal(loc=128, scale=10) 
        penalty = -abs(simulated_flips - 128)
        total_penalty += penalty
        
    avg_penalty = total_penalty / num_samples
    
    # Reward is the inverse of the penalty (closer to 0 penalty is better)
    reward = avg_penalty
    return reward
