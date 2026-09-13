"""Rejoue la logique des notebooks 01 et 07 (hors tracés) pour détecter les régressions."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing_agent"))

import numpy as np
import pandas as pd

from gym_wrapper import STATE_BINS, TicketEnv, discretize_observation
from market import TicketMarketModel


def nb01_like():
    market = TicketMarketModel(artist="Powerwolf", venue="Komplex 457", horizon=120, seed=1)
    market.current_price = 55.0
    rows = []
    while not market.done:
        market.step()
        active = sum(1 for b in market.buyers if b.visits_made > 0 and not b.has_bought)
        rows.append(
            {
                "t": market.initial_time - market.time_remaining,
                "visits": market.expected_visits_by_step[market._step_index - 1],
                "sold_step": market.tickets_sold_this_step,
                "sold_cum": market.tickets_sold_total,
                "active": active,
            }
        )
    df = pd.DataFrame(rows)
    assert 1 <= len(df) <= 120                       # peut vendre tout le stock avant la date
    assert df["sold_cum"].is_monotonic_increasing
    print(f"  nb01: {market.artist.name} vendus={market.tickets_sold_total}/{market.initial_tickets} "
          f"fill={market.fill_rate:.1%}")


def nb07_like(episodes=400):
    env = TicketEnv(horizon=90)
    n_actions = env.action_space.n
    q = np.zeros((*STATE_BINS, n_actions))
    alpha, gamma, eps = 0.1, 0.98, 1.0
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        s = discretize_observation(obs)
        done = False
        while not done:
            a = int(rng.integers(n_actions)) if rng.random() < eps else int(np.argmax(q[s]))
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ns = discretize_observation(obs)
            best_next = 0.0 if done else np.max(q[ns])
            q[s][a] += alpha * (r + gamma * best_next - q[s][a])
            s = ns
        eps = max(0.05, eps * 0.99)
    visited = int(np.count_nonzero(q.any(axis=-1)))
    assert visited > 20
    print(f"  nb07: {episodes} épisodes, {visited}/{int(np.prod(STATE_BINS))} états visités")

    from agent_fixed import FixedPriceAgent
    from agent_qtable import QTableAgent
    from evaluate import compare_agents, summarize

    agents = [FixedPriceAgent.from_price(60), FixedPriceAgent.from_price(90), QTableAgent(q)]
    agents[1].name = "fixed_90"
    res = compare_agents(agents, lambda: TicketEnv(horizon=90), n_episodes=60)
    s = summarize(res)
    print(s.to_string())
    assert set(res["agent"]) == {"fixed", "fixed_90", "qtable"}


if __name__ == "__main__":
    print("- nb01_like"); nb01_like()
    print("- nb07_like"); nb07_like()
    print("\nOK")
