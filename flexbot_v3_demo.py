"""
FlexBot V3: Improved Robot Arm Simulation with Better Training
================================================================

Improvements over V2:
1. Better reward shaping (dense rewards)
2. Improved hyperparameters
3. Longer training with more stable learning
4. Better exploration strategy
5. Separate figure generation for report

Author: Seymur Hasanov (FlexBot Exploration for Capstone)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ============================================================================
# Improved Configuration
# ============================================================================
N_SEGMENTS = 3  # Simpler arm for better learning
BASE_LENGTH = 0.3
N_EPISODES = 2000  # More training
MAX_STEPS = 100  # Shorter episodes for faster learning
LEARNING_RATE = 1e-4  # Lower LR for stability
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02  # More exploration
N_UPDATE_EPOCHS = 10  # More optimization steps
BATCH_SIZE = 128  # Larger batches
UPDATE_FREQ = 5  # Update every 5 episodes

print("="*70)
print("FlexBot V3: Improved Robot Arm with Better Training")
print("="*70)

# ============================================================================
# Improved Robot Arm Environment
# ============================================================================
class RobotArmEnvV3:
    """Enhanced environment with better reward shaping."""
    
    def __init__(self, n_segments=3, segment_length=0.3):
        self.n_segments = n_segments
        self.base_length = segment_length
        self.difficulty = 0.0
        self.reset()
    
    def set_difficulty(self, difficulty):
        self.difficulty = np.clip(difficulty, 0, 1)
    
    def reset(self):
        # Less aggressive randomization initially
        variation = 0.05 + 0.15 * self.difficulty  # 5% to 20%
        self.segment_lengths = self.base_length * np.random.uniform(
            1 - variation, 1 + variation, self.n_segments
        )
        
        # Physical properties
        self.segment_masses = 1.0 + 0.3 * self.difficulty * np.random.uniform(-1, 1, self.n_segments)
        self.friction = 1.0 + 0.1 * self.difficulty * np.random.uniform(-1, 1)
        
        self.max_reach = np.sum(self.segment_lengths) * 0.95
        
        # Start near zero (easier to learn)
        self.joint_angles = np.random.uniform(-0.1, 0.1, self.n_segments)
        self.joint_velocities = np.zeros(self.n_segments)
        
        # Target in reachable workspace (start easier)
        min_radius = 0.2 + 0.2 * self.difficulty
        max_radius = min(0.5 + 0.3 * self.difficulty, self.max_reach * 0.9)
        angle = np.random.uniform(np.pi/6, 5*np.pi/6)  # Front hemisphere
        radius = np.random.uniform(min_radius, max_radius)
        self.target = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        
        self.steps = 0
        self.prev_distance = np.linalg.norm(self._forward_kinematics()[0] - self.target)
        self.initial_distance = self.prev_distance
        
        return self._get_state()
    
    def _forward_kinematics(self):
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
        end_effector, _ = self._forward_kinematics()
        
        # Normalized direction to target
        direction = self.target - end_effector
        distance = np.linalg.norm(direction)
        direction_normalized = direction / (distance + 1e-8)
        
        state = np.concatenate([
            np.sin(self.joint_angles),
            np.cos(self.joint_angles),
            self.joint_velocities * 5,  # Scale velocities
            end_effector / self.max_reach,  # Normalized position
            self.target / self.max_reach,  # Normalized target
            direction_normalized,  # Direction to target
            [distance / self.max_reach]  # Normalized distance
        ])
        
        # Small observation noise
        noise_std = 0.005 * self.difficulty
        return state + np.random.normal(0, noise_std, state.shape)
    
    def step(self, action):
        action = np.clip(action, -1, 1) * 0.1
        
        # Action noise
        action_noise_std = 0.01 * self.difficulty
        action = action + np.random.normal(0, action_noise_std, action.shape)
        
        # Smooth dynamics
        self.joint_velocities = 0.8 * self.joint_velocities + 0.2 * action
        self.joint_angles += self.joint_velocities
        self.joint_angles = np.clip(self.joint_angles, -np.pi * 0.8, np.pi * 0.8)
        
        end_effector, _ = self._forward_kinematics()
        distance = np.linalg.norm(end_effector - self.target)
        
        # IMPROVED REWARD SHAPING
        # 1. Distance reward (scaled)
        distance_reward = -distance * 2
        
        # 2. Progress reward (getting closer is good)
        progress = self.prev_distance - distance
        progress_reward = progress * 20  # Strong progress signal
        
        # 3. Success bonus
        success = distance < 0.04
        success_reward = 50.0 if success else 0.0
        
        # 4. Near-success bonus
        if distance < 0.1:
            near_bonus = (0.1 - distance) * 10
        else:
            near_bonus = 0
        
        # 5. Small action penalty (encourage efficiency)
        action_penalty = -0.01 * np.sum(action ** 2)
        
        # 6. Time penalty (encourage faster completion)
        time_penalty = -0.01
        
        reward = distance_reward + progress_reward + success_reward + near_bonus + action_penalty + time_penalty
        
        self.prev_distance = distance
        self.steps += 1
        
        done = success or self.steps >= MAX_STEPS
        
        info = {
            'distance': distance,
            'success': success,
            'progress': progress,
            'improvement': 1 - (distance / self.initial_distance)
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        ax.clear()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.2, 1.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Workspace
        theta = np.linspace(0, np.pi, 100)
        ax.plot(self.max_reach * np.cos(theta), self.max_reach * np.sin(theta), 
                'k--', alpha=0.2)
        
        _, positions = self._forward_kinematics()
        
        # Draw arm
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=6, solid_capstyle='round')
            ax.plot(x1, y1, 'ko', markersize=10)
        
        ax.plot(positions[-1][0], positions[-1][1], 'go', markersize=14, 
                label='End-effector', zorder=5)
        ax.plot(self.target[0], self.target[1], 'r*', markersize=18, 
                label='Target')
        
        ax.plot(0, 0, 'ks', markersize=12)
        ax.axhline(y=0, color='gray', linewidth=3)
        
        distance = np.linalg.norm(np.array(positions[-1]) - self.target)
        ax.set_title(title or f'Distance: {distance:.3f}')
        ax.legend(loc='upper right')
        
        return ax

# ============================================================================
# Improved Actor-Critic Network
# ============================================================================
class ActorCriticV3(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCriticV3, self).__init__()
        
        # Deeper network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        # Initialize log_std to encourage exploration
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(torch.clamp(self.actor_log_std, -2, 0.5))
        value = self.critic(features)
        return action_mean, action_std.expand_as(action_mean), value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, value, None
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        return action, value, log_prob
    
    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, value, entropy

# ============================================================================
# Improved PPO Training
# ============================================================================
def compute_gae(rewards, values, next_value, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return advantages

def train_ppo_v3(env, model, optimizer, scheduler, n_episodes=2000):
    """Improved PPO training with better stability."""
    
    reward_history = []
    success_history = []
    distance_history = []
    eval_results = []
    
    # Buffers
    states_buf = []
    actions_buf = []
    logprobs_buf = []
    values_buf = []
    rewards_buf = []
    dones_buf = []
    advantages_buf = []
    
    print("\nTraining with Improved PPO...")
    print("-" * 70)
    
    best_success_rate = 0
    
    for episode in range(n_episodes):
        # Slower curriculum
        difficulty = min(episode / (n_episodes * 0.8), 1.0)
        env.set_difficulty(difficulty)
        
        state = env.reset()
        episode_reward = 0
        ep_states, ep_actions, ep_logprobs, ep_values, ep_rewards, ep_dones = [], [], [], [], [], []
        
        for step in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action, value, log_prob = model.get_action(state_tensor)
            
            action_np = action.numpy().flatten()
            next_state, reward, done, info = env.step(action_np)
            
            ep_states.append(state)
            ep_actions.append(action_np)
            ep_logprobs.append(log_prob.item())
            ep_values.append(value.item())
            ep_rewards.append(reward)
            ep_dones.append(float(done))
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Compute advantages
        with torch.no_grad():
            next_value = model(torch.FloatTensor(state).unsqueeze(0))[2].item()
        
        advantages = compute_gae(ep_rewards, ep_values, next_value, ep_dones)
        returns = [adv + val for adv, val in zip(advantages, ep_values)]
        
        # Add to buffers
        states_buf.extend(ep_states)
        actions_buf.extend(ep_actions)
        logprobs_buf.extend(ep_logprobs)
        values_buf.extend(returns)
        rewards_buf.extend(ep_rewards)
        dones_buf.extend(ep_dones)
        advantages_buf.extend(advantages)
        
        # Track metrics
        reward_history.append(episode_reward)
        success_history.append(1 if info['success'] else 0)
        distance_history.append(info['distance'])
        
        # PPO update
        if (episode + 1) % UPDATE_FREQ == 0 and len(states_buf) >= BATCH_SIZE:
            states_t = torch.FloatTensor(np.array(states_buf))
            actions_t = torch.FloatTensor(np.array(actions_buf))
            old_logprobs_t = torch.FloatTensor(logprobs_buf).unsqueeze(1)
            returns_t = torch.FloatTensor(values_buf).unsqueeze(1)
            advantages_t = torch.FloatTensor(advantages_buf).unsqueeze(1)
            
            # Normalize advantages
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
            # Multiple PPO epochs
            for _ in range(N_UPDATE_EPOCHS):
                indices = np.random.permutation(len(states_buf))
                
                for start in range(0, len(states_buf) - BATCH_SIZE + 1, BATCH_SIZE):
                    batch_idx = indices[start:start + BATCH_SIZE]
                    
                    b_states = states_t[batch_idx]
                    b_actions = actions_t[batch_idx]
                    b_old_logprobs = old_logprobs_t[batch_idx]
                    b_returns = returns_t[batch_idx]
                    b_advantages = advantages_t[batch_idx]
                    
                    log_probs, values, entropy = model.evaluate(b_states, b_actions)
                    
                    ratio = torch.exp(log_probs - b_old_logprobs)
                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * (b_returns - values).pow(2).mean()
                    entropy_loss = -entropy.mean()
                    
                    loss = actor_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
            
            scheduler.step()
            
            # Clear buffers
            states_buf, actions_buf, logprobs_buf, values_buf = [], [], [], []
            rewards_buf, dones_buf, advantages_buf = [], [], []
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            success_rate = np.mean(success_history[-100:]) * 100
            avg_distance = np.mean(distance_history[-100:])
            
            # Evaluation
            eval_success = evaluate_policy_v3(env, model, n_episodes=20)
            eval_results.append((episode + 1, eval_success))
            
            if eval_success > best_success_rate:
                best_success_rate = eval_success
            
            print(f"Ep {episode+1:4d} | Reward: {avg_reward:7.1f} | "
                  f"Success: {success_rate:5.1f}% | Eval: {eval_success:5.1f}% | "
                  f"Dist: {avg_distance:.3f} | Diff: {difficulty:.2f}")
    
    return reward_history, success_history, distance_history, eval_results

def evaluate_policy_v3(env, model, n_episodes=20):
    successes = 0
    old_diff = env.difficulty
    env.set_difficulty(1.0)  # Always evaluate at max difficulty
    
    for _ in range(n_episodes):
        state = env.reset()
        
        for _ in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, deterministic=True)
            state, _, done, info = env.step(action.numpy().flatten())
            if done:
                break
        
        if info['success']:
            successes += 1
    
    env.set_difficulty(old_diff)
    return (successes / n_episodes) * 100

# ============================================================================
# Separate Figure Generation for Report
# ============================================================================
def save_separate_figures(env, model, reward_history, success_history, distance_history, eval_results, save_path):
    """Generate separate figures for technical report."""
    
    # 1. Training Reward Curve
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    window = 50
    smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, 'b-', linewidth=2, label='Episode Reward (smoothed)')
    ax1.fill_between(range(len(smoothed)), smoothed, alpha=0.3)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Training Reward Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f'{save_path}/fig_reward_curve.png', dpi=150)
    fig1.savefig(f'{save_path}/fig_reward_curve.pdf', dpi=150)
    plt.close(fig1)
    print("Saved: fig_reward_curve.png")
    
    # 2. Success Rate Curve
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    window = 100
    success_smooth = np.convolve(success_history, np.ones(window)/window, mode='valid')
    ax2.plot(success_smooth * 100, 'g-', linewidth=2, label='Training Success Rate')
    ax2.fill_between(range(len(success_smooth)), success_smooth * 100, alpha=0.3, color='green')
    
    if eval_results:
        episodes, rates = zip(*eval_results)
        ax2.plot(episodes, rates, 'ro-', markersize=8, linewidth=2, label='Evaluation Success Rate')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Task Success Rate Over Training', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f'{save_path}/fig_success_rate.png', dpi=150)
    fig2.savefig(f'{save_path}/fig_success_rate.pdf', dpi=150)
    plt.close(fig2)
    print("Saved: fig_success_rate.png")
    
    # 3. Distance to Target over Training
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    window = 100
    dist_smooth = np.convolve(distance_history, np.ones(window)/window, mode='valid')
    ax3.plot(dist_smooth, 'purple', linewidth=2)
    ax3.fill_between(range(len(dist_smooth)), dist_smooth, alpha=0.3, color='purple')
    ax3.axhline(y=0.04, color='r', linestyle='--', label='Success Threshold')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Final Distance to Target', fontsize=12)
    ax3.set_title('Average Distance to Target', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(f'{save_path}/fig_distance.png', dpi=150)
    fig3.savefig(f'{save_path}/fig_distance.pdf', dpi=150)
    plt.close(fig3)
    print("Saved: fig_distance.png")
    
    # 4. Robot Arm Demonstration (multiple successful reaches)
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))
    env.set_difficulty(1.0)
    
    for idx, ax in enumerate(axes4.flat):
        state = env.reset()
        trajectory = []
        
        for _ in range(MAX_STEPS):
            end_eff, _ = env._forward_kinematics()
            trajectory.append(end_eff.copy())
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, deterministic=True)
            state, _, done, info = env.step(action.numpy().flatten())
            if done:
                break
        
        env.render(ax)
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'g--', alpha=0.6, linewidth=2)
        status = "✓ Success" if info['success'] else f"d={info['distance']:.3f}"
        ax.set_title(f'Episode {idx+1}: {status}', fontsize=11)
    
    fig4.suptitle('Policy Demonstrations at Maximum Difficulty', fontsize=14, fontweight='bold')
    fig4.tight_layout()
    fig4.savefig(f'{save_path}/fig_demonstrations.png', dpi=150)
    fig4.savefig(f'{save_path}/fig_demonstrations.pdf', dpi=150)
    plt.close(fig4)
    print("Saved: fig_demonstrations.png")
    
    # 5. Domain Randomization Visualization
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    ax5.set_xlim(-1.1, 1.1)
    ax5.set_ylim(-0.2, 1.3)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    
    for i in range(20):
        env_rand = RobotArmEnvV3()
        env_rand.set_difficulty(1.0)
        env_rand.reset()
        
        # Draw workspace boundary
        theta = np.linspace(0, np.pi, 50)
        ax5.plot(env_rand.max_reach * np.cos(theta), 
                 env_rand.max_reach * np.sin(theta), 
                 'b-', alpha=0.1, linewidth=1)
        
        # Draw target
        ax5.plot(env_rand.target[0], env_rand.target[1], 'r*', markersize=12, alpha=0.7)
    
    ax5.axhline(y=0, color='gray', linewidth=3)
    ax5.plot(0, 0, 'ks', markersize=12)
    ax5.set_title('Domain Randomization: 20 Different Scenarios', fontsize=14, fontweight='bold')
    ax5.set_xlabel('X Position')
    ax5.set_ylabel('Y Position')
    fig5.tight_layout()
    fig5.savefig(f'{save_path}/fig_domain_randomization.png', dpi=150)
    fig5.savefig(f'{save_path}/fig_domain_randomization.pdf', dpi=150)
    plt.close(fig5)
    print("Saved: fig_domain_randomization.png")
    
    # 6. End-Effector Trajectories
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    ax6.set_xlim(-1.1, 1.1)
    ax6.set_ylim(-0.2, 1.3)
    ax6.set_aspect('equal')
    
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    successes = 0
    
    for i in range(10):
        env.set_difficulty(1.0)
        state = env.reset()
        trajectory = []
        
        for _ in range(MAX_STEPS):
            end_eff, _ = env._forward_kinematics()
            trajectory.append(end_eff.copy())
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, deterministic=True)
            state, _, done, info = env.step(action.numpy().flatten())
            if done:
                break
        
        if info['success']:
            successes += 1
        
        trajectory = np.array(trajectory)
        ax6.plot(trajectory[:, 0], trajectory[:, 1], '-', color=colors[i], 
                 alpha=0.8, linewidth=2)
        ax6.plot(env.target[0], env.target[1], '*', color=colors[i], markersize=15)
        ax6.plot(trajectory[-1, 0], trajectory[-1, 1], 'o', color=colors[i], markersize=8)
    
    ax6.plot(0, 0, 'ks', markersize=12)
    ax6.axhline(y=0, color='gray', linewidth=3)
    ax6.grid(True, alpha=0.3)
    ax6.set_title(f'End-Effector Trajectories ({successes}/10 successful)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('X Position')
    ax6.set_ylabel('Y Position')
    fig6.tight_layout()
    fig6.savefig(f'{save_path}/fig_trajectories.png', dpi=150)
    fig6.savefig(f'{save_path}/fig_trajectories.pdf', dpi=150)
    plt.close(fig6)
    print("Saved: fig_trajectories.png")
    
    return successes

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    
    env = RobotArmEnvV3(n_segments=N_SEGMENTS, segment_length=BASE_LENGTH)
    
    # State: sin + cos + velocities + pos + target + direction + distance
    state_dim = N_SEGMENTS * 3 + 2 + 2 + 2 + 1
    action_dim = N_SEGMENTS
    
    print(f"\nEnvironment: {N_SEGMENTS}-segment arm")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Training: {N_EPISODES} episodes")
    
    model = ActorCriticV3(state_dim, action_dim, hidden_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    
    # Train
    reward_history, success_history, distance_history, eval_results = train_ppo_v3(
        env, model, optimizer, scheduler, n_episodes=N_EPISODES
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    # Final evaluation
    print("\nFinal Evaluation (50 episodes at max difficulty)...")
    final_success = evaluate_policy_v3(env, model, n_episodes=50)
    print(f"Final Success Rate: {final_success:.1f}%")
    
    # Save separate figures
    save_path = "/Users/seymurhasanov/Desktop/Harvard HES/CSCI E597/FlexBot_exploration/figures"
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\nGenerating figures for report...")
    demo_success = save_separate_figures(env, model, reward_history, success_history, 
                                         distance_history, eval_results, save_path)
    
    print("\n" + "="*70)
    print("FLEXBOT V3 COMPLETE!")
    print(f"Final Success Rate: {final_success:.1f}%")
    print(f"Demo Success: {demo_success}/10")
    print("="*70)
