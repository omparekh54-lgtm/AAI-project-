from __future__ import annotations
import argparse
import os
import torch
from .environment.simulator import CabDispatchEnv
from .agents.dqn import SharedDQNAgent


def valid_actions(env, cab):
    actions = [env.ACTION_WAIT, env.ACTION_ACCEPT]
    x, y = cab.position
    if x > 0: actions.append(env.ACTION_NORTH)
    if x < 4: actions.append(env.ACTION_SOUTH)
    if y > 0: actions.append(env.ACTION_WEST)
    if y < 4: actions.append(env.ACTION_EAST)
    return actions


def train(episodes=100, seed=42):
    agent = SharedDQNAgent(gamma=0.95, seed=seed)
    rewards = []
    for episode in range(1, episodes + 1):
        env = CabDispatchEnv(seed=seed + episode, fleet_size=8, day_minutes=180)
        env.reset()
        episode_reward = 0.0
        for _ in range(env.day_minutes):
            before_completed, before_cancelled, before_empty = env.completed, env.cancelled, env.total_empty_distance
            states = env.observations()
            actions = {}
            for cab in env.cabs:
                if cab.status == "idle":
                    actions[cab.cab_id] = agent.act(states[cab.cab_id], valid_actions(env, cab))
                else:
                    actions[cab.cab_id] = env.ACTION_WAIT
            env.step(actions)
            next_states = env.observations()
            reward = 4.0 * (env.completed - before_completed) - 5.0 * (env.cancelled - before_cancelled) - 0.08 * (env.total_empty_distance - before_empty)
            episode_reward += reward
            done = env.minute >= env.day_minutes
            for cab in env.cabs:
                s = states[cab.cab_id]
                ns = next_states[cab.cab_id]
                agent.remember(s, actions[cab.cab_id], reward / max(1, len(env.cabs)), ns, done)
            agent.learn()
        rewards.append(episode_reward)
        if episode % max(1, episodes // 10) == 0:
            print(f"Episode {episode:4d}/{episodes} | reward={episode_reward:8.2f} | epsilon={agent.epsilon:.3f}")
    os.makedirs("models", exist_ok=True)
    torch.save({"model": agent.policy.state_dict(), "epsilon": agent.epsilon, "rewards": rewards}, "models/dqn_latest.pt")
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.episodes, args.seed)
