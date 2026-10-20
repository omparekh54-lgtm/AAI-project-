"""Train the VDN extension (centralised training, decentralised execution).

Local Q-networks share weights; the joint team value is the sum of the chosen
local Q-values over all cabs (busy cabs contribute Q(s, WAIT)). Epsilon decays
over training episodes and off-shift minutes are not learned from.
"""
from __future__ import annotations
import argparse
import json
import os
import torch
from .agents.vdn import VDNAgent
from .environment.simulator import CabDispatchEnv
from .train import STATE_SIZE, FLEET_CURRICULUM


def train(episodes=240, seed=42, day_minutes=1440, learn_every=4, fleets=FLEET_CURRICULUM, out_dir='models'):
    # one VDN network handles any fleet size because weights are shared across cabs
    agent = VDNAgent(n_agents=8, state_size=STATE_SIZE, seed=seed, batch_size=32)
    rewards, losses = [], []
    for episode in range(1, episodes + 1):
        fleet = fleets[(episode - 1) % len(fleets)]
        env = CabDispatchEnv(seed=seed + episode, fleet_size=fleet, day_minutes=day_minutes, dispatch_reach=True)
        states = env.reset()
        agent.epsilon = max(agent.epsilon_min, 1.0 - (episode - 1) / max(1, episodes * 0.7))
        total_reward, episode_losses = 0.0, []
        for step in range(day_minutes):
            ordered = [states[i] for i in range(fleet)]
            valid = [env.valid_actions(c) if c.status == 'idle' else [env.ACTION_WAIT] for c in env.cabs]
            chosen = agent.act(ordered, valid)
            env.step({i: chosen[i] for i in range(fleet)})
            nxt = env.observations()
            next_valid = [env.valid_actions(c) if c.status == 'idle' else [env.ACTION_WAIT] for c in env.cabs]
            step_rewards = [env.last_rewards[i] for i in range(fleet)]
            total_reward += sum(step_rewards)
            # Off-shift, with every cab idle, nothing can be decided or learned.
            if env.is_shift_active or any(c.status != 'idle' for c in env.cabs):
                agent.remember(ordered, chosen, step_rewards, [nxt[i] for i in range(fleet)], next_valid,
                               step == day_minutes - 1)
                if step % learn_every == 0:
                    loss = agent.learn()
                    if loss is not None:
                        episode_losses.append(loss)
            states = nxt
        rewards.append(float(total_reward / fleet))
        losses.append(float(sum(episode_losses) / max(1, len(episode_losses))))
        if episode % max(1, episodes // 12) == 0:
            print(f'[vdn s{seed}] episode {episode}/{episodes} fleet={fleet} reward/cab={rewards[-1]:.1f} eps={agent.epsilon:.2f}', flush=True)
    os.makedirs(out_dir, exist_ok=True)
    path = f'{out_dir}/vdn_s{seed}.pt'
    torch.save({'model': agent.policy.state_dict(), 'rewards': rewards, 'episodes': episodes, 'seed': seed,
                'state_size': STATE_SIZE, 'mode': 'vdn', 'simulator_version': 3}, path)
    with open(f'results/training_curve_vdn_s{seed}.json', 'w', encoding='utf-8') as handle:
        json.dump({'mode': 'vdn', 'episodes': episodes, 'seed': seed, 'rewards': rewards, 'losses': losses}, handle)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=240)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    train(a.episodes, a.seed)
