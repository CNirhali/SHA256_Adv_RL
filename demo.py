import torch
import sys
import os

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.env.crypto_blender_env import CryptoBlenderEnv
from src.agent.ppo_network import PPOActorCritic
from src.evaluation.avalanche_scorer import evaluate_avalanche_criterion

def main():
    print("--- AI-Driven Post-Quantum Hash Function Demo ---")
    
    max_steps = 10
    
    print("\n1. Initializing Environment...")
    env = CryptoBlenderEnv(max_steps=max_steps)
    obs, _ = env.reset()
    print(f"   Environment step sequences to build: {max_steps}")
    
    print("\n2. Initializing PPO Agent...")
    action_dims = [env.num_ops, env.num_registers, env.num_registers, env.num_registers]
    agent = PPOActorCritic(max_steps=max_steps, action_dims=action_dims)
    print("   Agent Network Initialized.")
    
    print("\n3. Generating Candidate Hash Architecture...")
    for step in range(max_steps):
        # The observation must be formatted as float tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Get action from the policy network
        with torch.no_grad():
            action_tensor, log_probs, value = agent.get_action(obs_tensor)
            
        action = action_tensor.squeeze(0).numpy()
        
        # Take step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        op_names = {0: "ADD", 1: "XOR", 2: "ROTR", 3: "CH", 4: "MAJ"}
        op_name = op_names.get(action[0], str(action[0]))
        
        print(f"   Step {step + 1:02d}: Op={op_name}, Tgt=R{action[1]}, Src1=R{action[2]}, Src2=R{action[3]}")
        
        if terminated or truncated:
            break
            
    print("\n4. Architecture Construction Complete.")
    final_architecture = env.architecture_graph
    
    print("\n5. Evaluating Architecture with Strict Avalanche Criterion (SAC)...")
    # Simulate avalanche criterion evaluating the architecture
    reward = evaluate_avalanche_criterion(final_architecture, num_samples=100)
    print(f"   Evaluation Complete.")
    print(f"   -> Calculated SAC Reward: {reward:.4f} (Higher is closer to ideal 128 bit-flips)")

if __name__ == "__main__":
    main()
