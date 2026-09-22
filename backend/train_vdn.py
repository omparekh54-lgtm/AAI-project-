from __future__ import annotations
import argparse
import json
import os
import torch
from .agents.vdn import VDNAgent
from .environment.simulator import CabDispatchEnv


def train(episodes=120, seed=42, day_minutes=1440, fleet_size=8, learn_every=4):
    agent = VDNAgent(n_agents=fleet_size, seed=seed)
    rewards, losses = [], []
    for episode in range(1, episodes + 1):
        env = CabDispatchEnv(seed=seed + episode, fleet_size=fleet_size, day_minutes=day_minutes)
        states = env.reset()
        total_reward, episode_losses = 0.0, []
        for step in range(day_minutes):
            ordered_states = [states[i] for i in range(fleet_size)]
            valid = [env.valid_actions(c) if c.status == 'idle' else [env.ACTION_WAIT] for c in env.cabs]
            actions_list = agent.act(ordered_states, valid)
            actions = {i: actions_list[i] for i in range(fleet_size)}
            env.step(actions)
            next_states = env.observations()
            next_valid = [env.valid_actions(c) if c.status == 'idle' else [env.ACTION_WAIT] for c in env.cabs]
            rewards_step = [env.last_rewards[i] for i in range(fleet_size)]
            total_reward += sum(rewards_step)
            agent.remember(ordered_states, actions_list, rewards_step,
                           [next_states[i] for i in range(fleet_size)], next_valid,
                           step == day_minutes - 1)
            if step % learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)
            states = next_states
        rewards.append(float(total_reward)); losses.append(float(sum(episode_losses) / max(1, len(episode_losses))))
        if episode % max(1, episodes // 10) == 0:
            print(f'VDN episode {episode}/{episodes} reward={total_reward:.2f} epsilon={agent.epsilon:.3f}')
    os.makedirs('models', exist_ok=True); os.makedirs('results', exist_ok=True)
    torch.save({'model': agent.policy.state_dict(), 'epsilon': agent.epsilon, 'rewards': rewards}, 'models/vdn_latest.pt')
    with open('results/vdn_training_curve.json', 'w', encoding='utf-8') as f:
        json.dump({'episodes': episodes, 'rewards': rewards, 'losses': losses}, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--episodes', type=int, default=120); p.add_argument('--seed', type=int, default=42); p.add_argument('--day-minutes', type=int, default=1440); p.add_argument('--fleet-size', type=int, default=8); p.add_argument('--learn-every', type=int, default=4)
    a = p.parse_args(); train(a.episodes, a.seed, a.day_minutes, a.fleet_size, a.learn_every)
