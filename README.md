# SHA256 Advanced RL (SHA256_Adv_RL)

**AI-Driven Post-Quantum Hash Function Architecture Generation**

This project leverages Deep Reinforcement Learning (Proximal Policy Optimization - PPO) to discover novel, highly-secure hash function architectures. By treating cryptographic construction as a sequential decision-making process, our RL agent generates sequences of cryptographic operations to optimize the Strict Avalanche Criterion (SAC).

## Overview

The core of the system models the hash function generation within a custom OpenAI Gymnasium environment (`CryptoBlenderEnv`). The agent learns to select a valid sequence of bitwise operations (e.g., `ADD`, `XOR`, `ROTR`, `CH`, `MAJ`) acting across a set of registers to maximize cryptographic diffusion and confusion.

### Key Components

*   **Environment (`src/env/crypto_blender_env.py`)**: A Gymnasium environment where actions map to inserting cryptographic operations into a growing architecture graph.
*   **Agent (`src/agent/ppo_network.py`)**: A PPO Actor-Critic neural network architecture designed to learn the optimal policy for operation sequencing.
*   **Evaluation (`src/evaluation/avalanche_scorer.py`)**: Scoring mechanism based on the Strict Avalanche Criterion (SAC) to grant rewards based on bit-flip characteristics.
*   **Cloud & Deployment (`src/cloud/*`)**: Includes a `Dockerfile` and `sagemaker_entrypoint.py` for large-scale training on AWS SageMaker.

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch
*   Gymnasium

Install the dependencies:
```bash
pip install -r requirements.txt
```

### Running the Demo

A demonstration script is provided to showcase the end-to-end pipeline of the environment initialization, agent action sampling, and architecture evaluation.

```bash
python demo.py
```

### Project Structure
```text
.
├── demo.py                        # Main execution script
├── requirements.txt               # Required packages
└── src/
    ├── agent/                     
    │   └── ppo_network.py         # PPO Agent implementation
    ├── env/
    │   ├── bitwise_ops.py         # Definitions of available cryptographic ops
    │   └── crypto_blender_env.py  # Custom Gym Environment
    ├── evaluation/
    │   └── avalanche_scorer.py    # SAC rewards computation
    └── cloud/
        ├── Dockerfile             # Container definition for cloud training
        └── sagemaker_entrypoint.py# Entry point for SageMaker deployment
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.