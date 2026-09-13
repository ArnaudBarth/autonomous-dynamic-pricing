"""Évaluation appariée : baselines vs Q-table vs DQN sur la randomisation de domaine.

    python scripts/run_comparison.py --episodes 500          # Q-table la plus récente
    python scripts/run_comparison.py --dqn                   # + DQN le plus récent
    python scripts/run_comparison.py --qtable-version 20260828-1815   # version précise
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing_agent"))

import pandas as pd

from gym_wrapper import TicketEnv
from agent_fixed import FixedPriceAgent
from agent_qtable import QTableAgent
from evaluate import compare_agents, summarize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--qtable-version", default=None, help="défaut : la plus récente")
    p.add_argument("--dqn", nargs="?", const="latest", default=None,
                   help="inclut le DQN (optionnel : version précise, défaut = plus récent)")
    args = p.parse_args()

    def _desc(agent):
        m = agent.meta
        return f"{agent.path.name}  ({m.get('created', '?')}, {m.get('episodes', '?')} épisodes)"

    qtable = QTableAgent.load(args.qtable_version)
    print(f"Q-table : {_desc(qtable)}")

    agents = [
        FixedPriceAgent.from_price(70),        # prix fixe "réaliste" (instinct : remplir la salle)
        FixedPriceAgent.from_price(90),        # prix fixe optimum "oracle" (inconnaissable a priori)
        qtable,
    ]
    agents[0].name = "fixed"
    agents[1].name = "fixed_90"

    if args.dqn:
        from agent_dqn import DQNAgent

        dqn = DQNAgent.load(None if args.dqn == "latest" else args.dqn)
        print(f"DQN     : {_desc(dqn)}")
        agents.append(dqn)

    results = compare_agents(
        agents, lambda: TicketEnv(horizon=args.horizon), n_episodes=args.episodes
    )

    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    print(summarize(results))

    wide = results.pivot_table(index="seed", columns="agent", values="ca")
    fixed_cols = [c for c in wide.columns if c.startswith("fixed")]
    ref = wide[fixed_cols].max(axis=1)     # meilleur prix fixe par scénario (borne haute des baselines)
    print("\nCA vs MEILLEUR prix fixe choisi par scénario :")
    for name in [a for a in ("qtable", "dqn") if a in wide.columns]:
        gain = (wide[name] - ref).mean() / ref.mean() * 100
        win = (wide[name] > ref).mean()
        print(f"  {name:10s}  CA {gain:+5.1f} %   gagne dans {win:.0%} des scénarios")


if __name__ == "__main__":
    main()
