from __future__ import annotations
import argparse
import json
import os
import torch
from .environment.simulator import CabDispatchEnv
from .agents.dqn import SharedDQNAgent


def train(episodes=360, seed=42, day_minutes=1440, fleet_size=8, learn_every=4):
    agent = SharedDQNAgent(gamma=0.95, seed=seed)
    rewards, losses, epsilons = [], [], []
    for episode in range(1, episodes + 1):
        env = CabDispatchEnv(seed=seed + episode, fleet_size=fleet_size, day_minutes=day_minutes)
        states = env.reset()
        episode_reward = 0.0
        episode_losses = []
        for step in range(day_minutes):
            actions = {}
            for cab in env.cabs:
                if cab.status == "idle":
                    actions[cab.cab_id] = agent.act(states[cab.cab_id], env.valid_actions(cab))
                else:
                    actions[cab.cab_id] = env.ACTION_WAIT
            env.step(actions)
            next_states = env.observations()
            done = step == day_minutes - 1
            for cab in env.cabs:
                reward = env.last_rewards[cab.cab_id]
                episode_reward += reward
                agent.remember(states[cab.cab_id], actions[cab.cab_id], reward,
                               next_states[cab.cab_id], done)
            if step % learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)
            states = next_states
        rewards.append(float(episode_reward))
        losses.append(float(sum(episode_losses) / max(1, len(episode_losses))))
        epsilons.append(float(agent.epsilon))
        if episode % max(1, episodes // 12) == 0:
            print(f"Episode {episode:4d}/{episodes} | reward={episode_reward:10.2f} | loss={losses[-1]:.4f} | epsilon={agent.epsilon:.3f}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.save({"model": agent.policy.state_dict(), "epsilon": agent.epsilon,
                "rewards": rewards, "losses": losses, "episodes": episodes},
               "models/dqn_latest.pt")
    with open("results/training_curve.json", "w", encoding="utf-8") as handle:
        json.dump({"episodes": episodes, "rewards": rewards, "losses": losses, "epsilons": epsilons}, handle, indent=2)
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--day-minutes", type=int, default=1440)
    parser.add_argument("--fleet-size", type=int, default=8)
    parser.add_argument("--learn-every", type=int, default=4)
    args = parser.parse_args()
    train(args.episodes, args.seed, args.day_minutes, args.fleet_size, args.learn_every)
