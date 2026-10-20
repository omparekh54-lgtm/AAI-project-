"""Execution policies built from trained agents.

* ``dqn_policy``            – every idle cab chooses accept / wait / move itself.
* ``HybridMatchingPolicy``  – the dispatcher assigns riders to idle cabs
  (joint matching, as in the slides: "accept the assigned ride"); cabs left
  without a ride use the learned DQN to decide where to wait or reposition.
"""
from __future__ import annotations
import torch
from backend.coordination.predictive import coordinated_matching


def dqn_policy(agent):
    def policy(env):
        states = env.observations()
        actions = {}
        for cab in env.cabs:
            if cab.status == 'idle':
                actions[cab.cab_id] = agent.act(states[cab.cab_id], env.valid_actions(cab), explore=False)
            else:
                actions[cab.cab_id] = env.ACTION_WAIT
        return actions
    return policy


def reposition_actions(env, cab):
    """Valid actions for a cab that has no assigned rider (ACCEPT removed)."""
    return [a for a in env.valid_actions(cab) if a != env.ACTION_ACCEPT]


class HybridMatchingPolicy:
    def __init__(self, agent, explore=False):
        self.agent = agent
        self.explore = explore

    def __call__(self, env):
        actions = coordinated_matching(env)
        states = env.observations()
        for cab in env.cabs:
            if cab.status == 'idle' and env.is_shift_active and actions[cab.cab_id] == env.ACTION_WAIT:
                actions[cab.cab_id] = self.agent.act(states[cab.cab_id], reposition_actions(env, cab), explore=self.explore)
        return actions


class VDNPolicy:
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, env):
        states = env.observations()
        valid = [env.valid_actions(c) if c.status == 'idle' else [env.ACTION_WAIT] for c in env.cabs]
        chosen = self.agent.act([states[c.cab_id] for c in env.cabs], valid, explore=False)
        return {c.cab_id: chosen[i] for i, c in enumerate(env.cabs)}
