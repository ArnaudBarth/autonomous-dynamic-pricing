"""Comparaison appariée des agents de tarification.

Chaque agent affronte **la même série de scénarios de randomisation de domaine**
(mêmes graines) : la comparaison est appariée, donc les écarts de revenu ne sont
pas dus à la chance du tirage d'artiste / de salle.
"""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd


def run_episode(agent, env, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    done = False
    steps = 0
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    ticket_rev = info.get("ticket_revenue_total", info["revenue_total"])
    ca = info.get("ca_total")                              # CA = billetterie + revenus annexes
    if ca is None:                                         # env antérieur au champ ca_total
        ca = ticket_rev + info.get("ancillary_revenue_total", 0.0)
    return {
        "agent": getattr(agent, "name", agent.__class__.__name__),
        "seed": seed,
        "revenue": info["revenue_total"],                 # marge de contribution (== objectif RL)
        "ca": ca,
        "ticket_revenue": ticket_rev,                     # billetterie brute
        "fill_rate": info["fill_rate"],
        "tickets_sold": info["tickets_sold_total"],
        "steps": steps,
        "artist": info["artist"],
        "venue": info["venue"],
        "popularity": round(info["popularity_index"], 2),
    }


def evaluate_agent(agent, env, n_episodes: int = 200, base_seed: int = 10_000) -> pd.DataFrame:
    return pd.DataFrame(run_episode(agent, env, base_seed + i) for i in range(n_episodes))


def compare_agents(
    agents: Iterable,
    env_factory: Callable[[], object],
    n_episodes: int = 200,
    base_seed: int = 10_000,
) -> pd.DataFrame:
    frames = []
    for agent in agents:
        frames.append(evaluate_agent(agent, env_factory(), n_episodes, base_seed))
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Marge, CA et taux de remplissage moyens par agent, + delta vs baseline 'fixed'.

    ``revenue`` = marge de contribution (objectif optimisé) ; ``ca`` = chiffre
    d'affaires (billetterie + revenus annexes). Les deux sont reportés.
    """
    agg = results.groupby("agent").agg(
        revenue_mean=("revenue", "mean"),
        ca_mean=("ca", "mean") if "ca" in results else ("revenue", "mean"),
        revenue_std=("revenue", "std"),
        fill_rate_mean=("fill_rate", "mean"),
        tickets_sold_mean=("tickets_sold", "mean"),
        n=("seed", "count"),
    )
    if "fixed" in agg.index:
        agg["revenue_vs_fixed_pct"] = (agg["revenue_mean"] / agg.loc["fixed", "revenue_mean"] - 1.0) * 100.0
        agg["ca_vs_fixed_pct"] = (agg["ca_mean"] / agg.loc["fixed", "ca_mean"] - 1.0) * 100.0
    return agg.sort_values("revenue_mean", ascending=False)
