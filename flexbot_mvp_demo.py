"""
FlexBot MVP Demo: Simulated Robot Arm Control with Domain Randomization
=========================================================================

This demo illustrates the core concepts of FlexBot:
1. Robot Arm Simulation — A 2D robot arm that reaches targets
2. Domain Randomization — Varying target positions, arm lengths, etc.
3. Policy Learning — Simple neural network learns to control the arm
4. Synthetic Data Generation — Generate many training scenarios

This is a simplified 2D planar arm, but the concepts scale to 3D robots
in Isaac Sim, MuJoCo, or PyBullet.

Author: Seymur Hasanov (FlexBot Exploration for Capstone)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# Configuration
# ============================================================================
N_SEGMENTS = 3  # Number of arm segments
BASE_LENGTH = 0.3  # Base length of each segment
WORKSPACE_SIZE = 1.0  # Workspace is [-1, 1] x [-1, 1]
N_EPISODES = 500
MAX_STEPS = 100
LEARNING_RATE = 0.001

print("="*60)
print("FlexBot MVP: Robot Arm Control with Domain Randomization")
print("="*60)

# ============================================================================
# 2D Planar Robot Arm Environment
# ============================================================================
class PlanarArmEnv:
    """
    A 2D planar robot arm with N segments.
    
    State: Joint angles (N values)
    Action: Joint angle changes (N values, clipped)
    Goal: Reach target position with end-effector
    """
    
    def __init__(self, n_segments=3, segment_length=0.3, randomize=True):
        self.n_segments = n_segments
        self.base_length = segment_length
        self.randomize = randomize
        self.reset()
    
    def reset(self):
        """Reset the environment with optional domain randomization."""
        # Joint angles (radians)
        self.joint_angles = np.random.uniform(-np.pi/4, np.pi/4, self.n_segments)
        
        # Domain Randomization
        if self.randomize:
            # Randomize segment lengths (±20%)
            self.segment_lengths = self.base_length * np.random.uniform(0.8, 1.2, self.n_segments)
            # Randomize target position
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0.3, 0.7)
            self.target = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        else:
            self.segment_lengths = np.ones(self.n_segments) * self.base_length
            self.target = np.array([0.5, 0.5])
        
        self.steps = 0
        return self._get_state()
    
    def _forward_kinematics(self):
        """Compute end-effector position from joint angles."""
        x, y = 0.0, 0.0
        angle = 0.0
        positions = [(x, y)]
        
        for i in range(self.n_segments):
            angle += self.joint_angles[i]
            x += self.segment_lengths[i] * np.cos(angle)
            y += self.segment_lengths[i] * np.sin(angle)
            positions.append((x, y))
        
        return np.array([x, y]), positions
    
    def _get_state(self):
        """Get observation: joint angles + end-effector pos + target pos."""
        end_effector, _ = self._forward_kinematics()
        return np.concatenate([
            self.joint_angles,  # Current joint angles
            end_effector,  # End-effector position
            self.target  # Target position
        ])
    
    def step(self, action):
        """Take action (joint angle changes) and return (state, reward, done, info)."""
        # Clip actions
        action = np.clip(action, -0.1, 0.1)
        
        # Apply action
        self.joint_angles += action
        self.joint_angles = np.clip(self.joint_angles, -np.pi, np.pi)
        
        # Compute reward
        end_effector, _ = self._forward_kinematics()
        distance = np.linalg.norm(end_effector - self.target)
        
        reward = -distance  # Negative distance as reward
        
        # Check if done
        self.steps += 1
        done = distance < 0.05 or self.steps >= MAX_STEPS
        
        info = {'distance': distance, 'success': distance < 0.05}
        
        return self._get_state(), reward, done, info
    
    def render(self, ax=None, show_target=True):
        """Render the arm."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Get joint positions
        _, positions = self._forward_kinematics()
        
        # Draw arm segments
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=4)
            ax.plot(x1, y1, 'ko', markersize=8)
        
        # Draw end-effector
        ax.plot(positions[-1][0], positions[-1][1], 'go', markersize=12, label='End-effector')
        
        # Draw target
        if show_target:
            ax.plot(self.target[0], self.target[1], 'r*', markersize=15, label='Target')
        
        ax.legend()
        ax.set_title(f'2D Robot Arm (Steps: {self.steps})')
        
        return ax

# ============================================================================
# Neural Network Policy
# ============================================================================
class PolicyNetwork(nn.Module):
    """Simple policy network: State -> Action."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1], scaled later
        )
    
    def forward(self, state):
        return self.network(state) * 0.1  # Scale to [-0.1, 0.1]

# ============================================================================
# Training Loop (Policy Gradient / REINFORCE-like)
# ============================================================================
def train_policy(env, policy, optimizer, n_episodes=500):
    """Train policy using simple policy gradient."""
    
    reward_history = []
    success_history = []
    
    print("\nTraining Policy with Domain Randomization...")
    print("-" * 40)
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_rewards = []
        saved_actions = []
        
        for step in range(MAX_STEPS):
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean = policy(state_tensor)
            
            # Sample action with exploration noise (Gaussian policy)
            std = 0.1
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            
            # Save action log probability for gradient
            saved_actions.append(dist.log_prob(action).sum())
            
            # Take step
            action_np = action.detach().numpy().flatten()
            next_state, reward, done, info = env.step(action_np)
            
            episode_rewards.append(reward)
            state = next_state
            
            if done:
                break
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        policy_loss = []
        for log_prob, G in zip(saved_actions, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_reward = sum(episode_rewards)
        reward_history.append(total_reward)
        success_history.append(1 if info['success'] else 0)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            success_rate = np.mean(success_history[-50:]) * 100
            print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:.3f} | Success Rate: {success_rate:.1f}%")
    
    return reward_history, success_history

# ============================================================================
# Visualization
# ============================================================================
def create_visualization(env, policy, reward_history, success_history):
    """Create comprehensive visualization of results."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('FlexBot MVP: Robot Arm Control with Domain Randomization\n' +
                 'Training a Neural Network to Control a Simulated Robot Arm',
                 fontsize=14, fontweight='bold')
    
    # 1. Robot arm reaching (before training)
    ax1 = fig.add_subplot(gs[0, 0])
    env_before = PlanarArmEnv(randomize=False)
    env_before.reset()
    env_before.render(ax1)
    ax1.set_title('Initial Arm Configuration')
    
    # 2. Robot arm reaching (after training - multiple samples)
    ax2 = fig.add_subplot(gs[0, 1])
    env_demo = PlanarArmEnv(randomize=True)
    state = env_demo.reset()
    for _ in range(50):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = policy(state_tensor).detach().numpy().flatten()
        state, _, done, _ = env_demo.step(action)
        if done:
            break
    env_demo.render(ax2)
    ax2.set_title('Trained Policy Reaching Target')
    
    # 3. Multiple targets (domain randomization demo)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Show multiple random targets
    for _ in range(20):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0.3, 0.7)
        target = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        ax3.plot(target[0], target[1], 'r*', markersize=10, alpha=0.7)
    ax3.set_title('Domain Randomization: Random Targets')
    ax3.plot(0, 0, 'ko', markersize=10, label='Base')
    ax3.legend()
    
    # 4. Training reward curve
    ax4 = fig.add_subplot(gs[1, 0])
    window = 20
    smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    ax4.plot(smoothed, 'b-', linewidth=1)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Training Reward (Moving Average)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Success rate over time
    ax5 = fig.add_subplot(gs[1, 1])
    window = 50
    success_smooth = np.convolve(success_history, np.ones(window)/window, mode='valid')
    ax5.plot(success_smooth * 100, 'g-', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title('Task Success Rate (50-episode average)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 100)
    
    # 6. End-effector trajectories
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(-1, 1)
    ax6.set_ylim(-1, 1)
    ax6.set_aspect('equal')
    
    # Run multiple episodes and plot trajectories
    for i in range(5):
        env_traj = PlanarArmEnv(randomize=True)
        state = env_traj.reset()
        trajectory = []
        
        for _ in range(50):
            end_eff, _ = env_traj._forward_kinematics()
            trajectory.append(end_eff.copy())
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy(state_tensor).detach().numpy().flatten()
            state, _, done, _ = env_traj.step(action)
            if done:
                break
        
        trajectory = np.array(trajectory)
        color = plt.cm.viridis(i / 5)
        ax6.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, alpha=0.7, linewidth=2)
        ax6.plot(env_traj.target[0], env_traj.target[1], '*', color=color, markersize=12)
    
    ax6.set_title('End-Effector Trajectories (5 episodes)')
    ax6.grid(True, alpha=0.3)
    ax6.plot(0, 0, 'ko', markersize=8)
    
    # 7. Domain randomization concept
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    
    concept_text = """
    DOMAIN RANDOMIZATION
    ====================
    
    Training with varied parameters:
    
    • Target positions: Random locations
    • Arm segment lengths: ±20% variation
    • Initial joint angles: Random
    
    Benefits:
    ✓ Better generalization
    ✓ Sim2Real transfer
    ✓ Robustness to noise
    """
    ax7.text(0.05, 0.95, concept_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 8. FlexBot pipeline
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    
    pipeline_text = """
    FLEXBOT PIPELINE
    ================
    
    ┌─────────────┐
    │ GenAI Scene │ ← Random environments
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Simulation │ ← Physics engine
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Training  │ ← Policy gradient
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Real Robot  │ ← Sim2Real transfer
    └─────────────┘
    """
    ax8.text(0.05, 0.95, pipeline_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 9. Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    final_success = np.mean(success_history[-50:]) * 100
    final_reward = np.mean(reward_history[-50:])
    
    summary_text = f"""
    RESULTS SUMMARY
    ===============
    
    Environment:
    • {N_SEGMENTS}-segment planar arm
    • Continuous action space
    • Domain randomization ON
    
    Training:
    • {N_EPISODES} episodes
    • Policy Gradient (REINFORCE)
    • Neural Network (2 hidden layers)
    
    Results:
    • Final Success Rate: {final_success:.1f}%
    • Final Avg Reward: {final_reward:.3f}
    
    This demonstrates:
    ✓ Synthetic data generation
    ✓ Domain randomization
    ✓ Policy learning
    """
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    return fig

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    
    # Create environment
    env = PlanarArmEnv(n_segments=N_SEGMENTS, segment_length=BASE_LENGTH, randomize=True)
    
    # State: joint angles (N) + end-effector (2) + target (2)
    state_dim = N_SEGMENTS + 2 + 2
    action_dim = N_SEGMENTS
    
    # Create policy
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=64)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Train
    reward_history, success_history = train_policy(env, policy, optimizer, n_episodes=N_EPISODES)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Create visualization
    fig = create_visualization(env, policy, reward_history, success_history)
    
    # Save results
    output_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/FlexBot_exploration/FlexBot_MVP_Results.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    
    pdf_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/FlexBot_exploration/FlexBot_MVP_Results.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("FLEXBOT MVP DEMO COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Domain randomization improves generalization")
    print("2. Neural networks can learn robot control from simulation")
    print("3. This scales to 3D robots in Isaac Sim / MuJoCo")
    print("="*60)
