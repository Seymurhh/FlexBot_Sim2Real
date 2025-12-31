# FlexBot: Sim2Real Robot Control with Domain Randomization

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An exploration of Sim2Real transfer techniques for robot manipulation using domain randomization, PPO reinforcement learning, and curriculum learning.**

## 📖 Overview

**FlexBot** addresses the "Data Wall" problem in Embodied AI: robots lack the equivalent of "the internet" that LLMs used for training. This project explores generating **synthetic training data** through simulation with **domain randomization** to train robot control policies that can transfer to real robots.

### Key Results
- **Final Success Rate**: 25% at maximum task difficulty
- **2.5× improvement** over baseline REINFORCE algorithm
- Maintains performance despite 30% variation in physical parameters
- **13-page technical report** with detailed analysis

## 🖼️ Results Visualization

### Training Progression
| Reward Curve | Success Rate |
|:------------:|:------------:|
| ![Reward](figures/fig_reward_curve.png) | ![Success](figures/fig_success_rate.png) |

### Robot Demonstrations
| Policy Demonstrations | End-Effector Trajectories |
|:---------------------:|:-------------------------:|
| ![Demos](figures/fig_demonstrations.png) | ![Trajectories](figures/fig_trajectories.png) |

### Domain Randomization
![Domain Randomization](figures/fig_domain_randomization.png)

## 🧮 Technical Approach

### Domain Randomization Parameters

| Parameter | Nominal | Range |
|-----------|---------|-------|
| Segment lengths | 0.25 m | ±30% |
| Segment masses | 1.0 kg | 0.5× – 1.5× |
| Friction | 1.0 | ±20% |
| Action noise | 0 | σ = 0.02 |
| Observation noise | 0 | σ = 0.01 |

### Algorithm: PPO + GAE + Curriculum Learning

```
PPO Objective: L = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]

where:
  r(θ) = π_new(a|s) / π_old(a|s)  (probability ratio)
  Â = GAE advantage estimate
  ε = 0.2 (clipping parameter)
```

### Curriculum Learning
- Difficulty increases from 0 → 1 over 70% of training
- Easy: Fixed arm lengths, nearby targets
- Hard: Full randomization, distant targets

### Training Challenges & Analysis

The 25% success rate reflects several interacting challenges documented in the technical report:

1. **Exploration-Exploitation Trade-off**: Stochastic policies hurt precision near targets
2. **Curriculum Learning Tension**: Previously-learned behaviors become suboptimal as difficulty increases
3. **Sparse Rewards**: Success only occurs when distance < 0.05
4. **High-Dimensional Randomization**: 6+ parameters create vast search space

**Proposed Solutions**: Hindsight Experience Replay (HER), Entropy Annealing, Automatic Domain Randomization (ADR)

## 📁 Repository Structure

```
FlexBot_Sim2Real/
├── README.md                         # This file
├── FlexBot_Technical_Report.pdf      # 13-page technical report
├── FlexBot_Technical_Report.tex      # LaTeX source
├── figures/                          # Separate figures for report
│   ├── fig_reward_curve.png          # Training reward progression
│   ├── fig_success_rate.png          # Success rate with curriculum
│   ├── fig_distance.png              # Distance to target
│   ├── fig_demonstrations.png        # Policy demos (6 episodes)
│   ├── fig_domain_randomization.png  # 25 randomized scenarios
│   └── fig_trajectories.png          # End-effector paths
├── flexbot_mvp_demo.py               # V1: Basic REINFORCE
├── flexbot_v2_demo.py                # V2: PPO + Domain Randomization
├── generate_figures.py               # Figure generation script
├── requirements.txt                  # Dependencies
└── LICENSE                           # MIT License
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Seymurhh/FlexBot_Sim2Real.git
cd FlexBot_Sim2Real

# Install dependencies
pip install -r requirements.txt

# Run V2 demo (PPO + Domain Randomization)
python flexbot_v2_demo.py

# Generate report figures
python generate_figures.py
```

## 📊 Training Results

| Episode | Avg Reward | Success Rate | Difficulty |
|---------|------------|--------------|------------|
| 100 | -85.94 | 12.0% | 0.14 |
| 500 | -72.26 | 10.0% | 0.71 |
| 700 | -65.26 | 20.0% | 1.00 |
| 1000 | -68.82 | 12.0% | 1.00 |
| **Eval** | **-56.70** | **25.0%** | 1.00 |

## 📚 Technical Report

A comprehensive **13-page technical report** is included covering:

- **Introduction**: Data Wall problem, Sim2Real motivation
- **Mathematical Framework**: Forward kinematics, PPO, GAE
- **Domain Randomization**: Parameters and visualization
- **Results Analysis**: Each figure explained with insights
- **Training Challenges**: Why performance plateaus
- **Future Work**: HER, AutoDR, real robot deployment

📄 **[View Technical Report (PDF)](FlexBot_Technical_Report.pdf)**

## 🔮 Future Directions

- [ ] Implement Hindsight Experience Replay (HER)
- [ ] Add visual observations (image-based control)
- [ ] Use Automatic Domain Randomization (ADR)
- [ ] Deploy on real robot (UR5, Franka Panda)
- [ ] Extend to 3D manipulation (6-DOF)
- [ ] Implement in NVIDIA Isaac Sim

## 📖 References

1. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
2. Tobin, J., et al. (2017). *Domain Randomization for Sim2Real Transfer.* IROS.
3. OpenAI (2019). *Learning Dexterous In-Hand Manipulation.* arXiv:1808.00177.
4. Andrychowicz, M., et al. (2017). *Hindsight Experience Replay.* NeurIPS.

## 👤 Author

**Seymur Hasanov**  
🔗 [LinkedIn](https://linkedin.com/in/seymurh) | [GitHub](https://github.com/Seymurhh)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---
