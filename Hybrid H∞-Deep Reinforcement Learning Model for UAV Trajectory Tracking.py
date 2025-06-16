import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import control as ctrl
import scipy.signal as signal
import gym
import matplotlib.pyplot as plt
import csv
import os
tf.keras.backend.set_floatx('float32')


class HinfTransferMap:
    """H∞ Transfer Map Optimization Module"""
    def __init__(self, order=8):
        self.order = order
        self.M = None  
    
    def optimize_transfer_map(self, G1, G2, W):
        P = ctrl.connect_SS(G1, G2, W)  # Simplified connection
        K, _, _ = ctrl.robust.hinfsyn(P, 1, 1)
        self.M = K  
    
    def discretize(self, dt=0.01):
        return ctrl.c2d(self.M, dt, method='tustin') if self.M else None
    
    def apply_transfer_map(self, u):
        if self.M is None:
            raise ValueError("Transfer map not optimized.")
        Md = self.discretize()
        num, den = Md.num[0][0], Md.den[0][0]
        return signal.lfilter(num, den, u)[0]

class AdvancedPPONetwork:
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], clip_ratio=0.1, ent_coef=0.01):
        self.actor = self._build_actor(state_dim, action_dim, hidden_dims)
        self.critic = self._build_critic(state_dim, hidden_dims)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef

    def _build_actor(self, state_dim, action_dim, hidden_dims):
        inputs = layers.Input(shape=(state_dim,))
        x = inputs
        for dim in hidden_dims:
            x = layers.Dense(dim, activation='tanh')(x)
        mu = layers.Dense(action_dim, activation='tanh')(x)
        log_std = tf.Variable(initial_value=-0.5 * np.ones(action_dim), trainable=True)
        return Model(inputs, mu), log_std

    def _build_critic(self, state_dim, hidden_dims):
        inputs = layers.Input(shape=(state_dim,))
        x = inputs
        for dim in hidden_dims:
            x = layers.Dense(dim, activation='tanh')(x)
        value = layers.Dense(1)(x)
        return Model(inputs, value)

    def get_action(self, state):
        state = np.expand_dims(state, axis=0) if state.ndim == 1 else state
        mu = self.actor[0](state)
        mu = tf.cast(mu, tf.float32)
        log_std = tf.cast(self.actor[1], tf.float32)  # ensure float32
        std = tf.exp(log_std)   

        action = mu + std * tf.random.normal(shape=mu.shape)
        log_prob = self.compute_log_prob(mu, std, action)

        return action.numpy()[0], log_prob.numpy()[0], mu.numpy()[0]

    def compute_log_prob(self, mu, std, actions):
        mu = tf.cast(mu, tf.float32)
        std = tf.cast(std, tf.float32)
        actions = tf.cast(actions, tf.float32)
        LOG_2PI = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)

        pre_sum = -0.5 * (((actions - mu) / std) ** 2 + 2 * tf.math.log(std) + LOG_2PI)

        return tf.reduce_sum(pre_sum, axis=1)

    def train(self, states, actions, returns, advantages, old_log_probs, epochs=10, batch_size=64):
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, returns, advantages, old_log_probs))
        dataset = dataset.shuffle(1024).batch(batch_size)

        for _ in range(epochs):
            for s, a, r, adv, logp_old in dataset:
                with tf.GradientTape() as tape:
                    mu = self.actor[0](s)
                    std = tf.exp(self.actor[1])
                    logp = self.compute_log_prob(mu, std, a)
                    ratio = tf.exp(logp - logp_old)

                    clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped * adv))

                    value_loss = tf.reduce_mean(tf.square(self.critic(s)[:, 0] - r))
                    entropy = tf.reduce_mean(-logp)

                    loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

                variables = self.actor[0].trainable_variables + self.critic.trainable_variables
                grads = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(grads, variables))

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

def collect_trajectories(env, model, horizon=2048, gamma=0.99, lam=0.95):
    states, actions, rewards, values, log_probs = [], [], [], [], []
    state = env.reset()
    
    for _ in range(horizon):
        action, logp, mu = model.get_action(state)
        value = model.critic(np.expand_dims(state, axis=0)).numpy()[0, 0]
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probs.append(logp)

        state = next_state
        if done:
            state = env.reset()

    values = np.append(values, model.critic(np.expand_dims(state, axis=0)).numpy()[0, 0])
    advantages, returns = compute_gae(rewards, values, gamma, lam)
    
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(returns, dtype=np.float32),
        np.array(advantages, dtype=np.float32),
        np.array(log_probs, dtype=np.float32),
        
    )

class HybridHinfDRL:
    def __init__(self, state_dim, action_dim):
        self.hinf_module = HinfTransferMap()
        self.ppo = AdvancedPPONetwork(state_dim, action_dim)
        self.lambda_mix = tf.Variable(0.5)

    def compute_action(self, state, reference, use_drl=True):
        hinf_action = self.hinf_module.apply_transfer_map(reference)
        if use_drl:
            drl_action, _, _ = self.ppo.get_action(state)
            return (1 - self.lambda_mix.numpy()) * hinf_action + self.lambda_mix.numpy() * drl_action
        return hinf_action

    def train_drl(self, env, episodes=100):
        episode_rewards = []

        for ep in range(episodes):
            s, a, r, adv, logp = collect_trajectories(env, self.ppo, gamma=0.98, lam=0.90)  # updated
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            self.ppo.train(s, a, r, adv, logp)

            avg_reward = np.sum(r) / len(r)
            episode_rewards.append(avg_reward)
            print(f"Episode {ep + 1}/{episodes} - Avg Reward: {avg_reward:.2f}")

        self._save_results(episode_rewards)
        return episode_rewards


    def _save_results(self, rewards, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)

        # Save plot
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label='Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Training Performance (PPO + H∞)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_rewards.png"))
        plt.close()

        # Save CSV
        with open(os.path.join(out_dir, "training_rewards.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Average Reward"])
            for i, reward in enumerate(rewards):
                writer.writerow([i + 1, reward])


class UAVTrajectoryTrackingEnv(gym.Env):
    """Gym-based UAV Trajectory Tracking Environment"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)  # ✅ Ensure float32
        return self.state

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        inertia = 0.9
        noise = np.random.normal(0, 0.01, size=self.state.shape).astype(np.float32)

        padded_action = np.zeros_like(self.state, dtype=np.float32)
        padded_action[:action.shape[0]] = action

        self.state = inertia * self.state + 0.1 * padded_action + noise
        reward = -np.linalg.norm(self.state).astype(np.float32)
        return self.state, reward, False, {}


# Example Usage
if __name__ == "__main__":
    env = UAVTrajectoryTrackingEnv(state_dim=12, action_dim=4)
    model = HybridHinfDRL(state_dim=12, action_dim=4)
    rewards = model.train_drl(env, episodes=100)
    print("Training completed, rewards:", rewards)
