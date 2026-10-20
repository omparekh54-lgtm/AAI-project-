"""Train parameter-shared DQN agents.

Modes
-----
full    Every idle cab chooses accept / wait / move on its own (decentralised).
hybrid  A dispatcher assigns riders; the DQN decides where unassigned idle
        cabs wait or reposition (RL for positioning, matching for assignment).

Learning uses decision steps only (semi-MDP): a transition starts when a cab
makes a choice and ends when it makes its next choice, with the discount
gamma**k for the k minutes in between. Busy cabs and off-shift minutes never
enter the replay buffer as fake "WAIT" decisions.
"""
from __future__ import annotations
import argparse
import json
import os
import torch
from .environment.simulator import CabDispatchEnv
from .agents.dqn import SharedDQNAgent
from .agents.policies import reposition_actions
from .coordination.predictive import coordinated_matching

STATE_SIZE = 22
FLEET_CURRICULUM = (8, 4, 8, 12, 8, 16)   # 8-cab pilot is trained most often


def train(episodes=240, seed=42, day_minutes=1440, mode='full', learn_every=4, fleets=FLEET_CURRICULUM,
          out_dir='models', quiet=False):
    agent = SharedDQNAgent(state_size=STATE_SIZE, gamma=0.95, seed=seed)
    rewards, losses, epsilons, used_fleets = [], [], [], []
    for episode in range(1, episodes + 1):
        fleet = fleets[(episode - 1) % len(fleets)]
        env = CabDispatchEnv(seed=seed + episode, fleet_size=fleet, day_minutes=day_minutes, dispatch_reach=True)
        env.reset()
        agent.epsilon = max(agent.epsilon_min, 1.0 - (episode - 1) / max(1, episodes * 0.7))
        pending = {}      # cab_id -> [state, action, discounted reward, discount]
        total_reward, episode_losses = 0.0, []

        def finalise(cab_id, next_state, valid, done):
            s, a, acc, disc = pending.pop(cab_id)
            agent.remember(s, a, acc, next_state, done, valid, disc)

        for step in range(day_minutes):
            states = env.observations()
            matching = coordinated_matching(env) if mode == 'hybrid' else {}
            actions = {}
            for cab in env.cabs:
                cid = cab.cab_id
                actions[cid] = env.ACTION_WAIT
                if cab.status != 'idle':
                    continue
                if not env.is_shift_active:
                    if cid in pending:
                        finalise(cid, states[cid], [env.ACTION_WAIT], False)
                    continue
                if mode == 'hybrid' and isinstance(matching.get(cid), tuple):
                    actions[cid] = matching[cid]        # assigned ride: not a learned decision
                    continue
                valid = reposition_actions(env, cab) if mode == 'hybrid' else env.valid_actions(cab)
                if cid in pending:
                    finalise(cid, states[cid], valid, False)
                action = agent.act(states[cid], valid)
                actions[cid] = action
                pending[cid] = [states[cid], action, 0.0, 1.0]
            env.step(actions)
            for cid, record in pending.items():
                reward = env.last_rewards[cid]
                record[2] += record[3] * reward
                record[3] *= agent.gamma
            total_reward += sum(env.last_rewards.values())
            if env.is_shift_active and step % learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)
        final_states = env.observations()
        for cid in list(pending):
            finalise(cid, final_states[cid], [env.ACTION_WAIT], True)
        rewards.append(float(total_reward / fleet))     # reward per cab so fleet sizes are comparable
        losses.append(float(sum(episode_losses) / max(1, len(episode_losses))))
        epsilons.append(float(agent.epsilon))
        used_fleets.append(fleet)
        if not quiet and episode % max(1, episodes // 12) == 0:
            print(f"[{mode} s{seed}] episode {episode:4d}/{episodes} fleet={fleet:2d} reward/cab={rewards[-1]:9.1f} "
                  f"loss={losses[-1]:.3f} eps={agent.epsilon:.2f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    path = f"{out_dir}/dqn_{mode}_s{seed}.pt"
    torch.save({'model': agent.policy.state_dict(), 'epsilon': agent.epsilon, 'rewards': rewards,
                'losses': losses, 'episodes': episodes, 'seed': seed, 'day_minutes': day_minutes,
                'fleet_sizes': list(fleets), 'mode': mode, 'state_size': STATE_SIZE,
                'simulator_version': 3}, path)
    with open(f"results/training_curve_{mode}_s{seed}.json", 'w', encoding='utf-8') as handle:
        json.dump({'mode': mode, 'episodes': episodes, 'day_minutes': day_minutes, 'seed': seed,
                   'fleet_sizes': used_fleets, 'rewards': rewards, 'losses': losses, 'epsilons': epsilons}, handle)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("full", "hybrid"), default="full")
    parser.add_argument("--day-minutes", type=int, default=1440)
    parser.add_argument("--learn-every", type=int, default=4)
    args = parser.parse_args()
    train(args.episodes, args.seed, args.day_minutes, args.mode, args.learn_every)
