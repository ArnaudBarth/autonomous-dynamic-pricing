"""Entraînement d'un agent tabulaire par **contrôle Monte-Carlo** (RL, phase 1).

Estimation de Q(état, action) *from scratch* avec NumPy sur l'environnement
discrétisé (temps × inventaire × popularité × rythme de vente × prix/référence ×
Fan Cost Index → action = palier de prix).

Méthode : contrôle Monte-Carlo on-policy à ε décroissant (Sutton & Barto, ch. 5).
Après **chaque épisode complet**, on remonte la trajectoire et on met chaque
paire (état, action) visitée à jour vers le **retour réellement observé**
``G_t = Σ γ^k r_{t+k}`` (pas de bootstrapping). Sur cet environnement à forte
variance inter-scénarios, la cible TD bootstrapée est trop instable ; le retour
Monte-Carlo est non biaisé. La cadence de décision de l'environnement
(``steps_per_decision``) ramène l'horizon à ~12 pas, ce qui borne la variance de G.

    python scripts/train_qtable.py --episodes 60000
    python scripts/train_qtable.py --episodes 60000 --resume   # reprend après une coupure

Le modèle final est sauvegardé **horodaté** (``q_table_AAAAMMJJ-HHMMSS.npy``) avec
un sidecar ``.meta.json``. Checkpoint tous les ``--checkpoint-every`` épisodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing_agent"))

from gym_wrapper import STATE_BINS, TicketEnv, discretize_observation
from registry import checkpoint_path, model_path, write_meta


def _verify(q_table: np.ndarray, horizon: int, n_episodes: int = 150) -> bool:
    """Contrôle de non-régression : la politique gloutonne bat-elle le prix fixe
    sur le chiffre d'affaires ? Un modèle entraîné sur un environnement périmé
    (kernel non redémarré, imports croisés) passe la validation de forme mais
    échoue ici. Retourne True si le CA moyen dépasse celui du prix fixe à 70.
    """
    from agent_fixed import FixedPriceAgent
    from agent_qtable import QTableAgent

    fixed = FixedPriceAgent.from_price(70.0)
    learned = QTableAgent(q_table)
    ca_fixed, ca_learned = 0.0, 0.0
    for i in range(n_episodes):
        for agent, bucket in ((fixed, "f"), (learned, "l")):
            env = TicketEnv(horizon=horizon)
            obs, _ = env.reset(seed=100_000 + i)
            done = False
            info: dict = {}
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(action)
                done = term or trunc
            if bucket == "f":
                ca_fixed += info["ca_total"]
            else:
                ca_learned += info["ca_total"]
    gain = ca_learned / ca_fixed - 1.0
    print(f"Verification ({n_episodes} scenarios apparies) : "
          f"CA vs prix fixe 70 = {gain:+.1%}")
    if gain <= 0:
        print("  [ATTENTION] le modele ne bat PAS le prix fixe -- probable "
              "environnement perime.\n  Relance dans un terminal neuf (pas depuis "
              "un kernel Jupyter) :  python scripts/train_qtable.py --episodes 60000")
    return gain > 0


def _atomic_save(path: str, array: np.ndarray, state: dict) -> None:
    """Écrit la Q-table + son état sans risque de fichier corrompu si coupure."""
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)                       # rename atomique
    with open(path + ".state.json", "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _load_checkpoint(path: str) -> tuple[np.ndarray, dict] | None:
    state_path = path + ".state.json"
    if not (os.path.exists(path) and os.path.exists(state_path)):
        return None
    with open(state_path, encoding="utf-8") as fh:
        return np.load(path), json.load(fh)


def train(
    episodes: int = 60_000,
    alpha: float = 0.05,               # pas d'apprentissage Monte-Carlo à α constant
    gamma: float = 0.97,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.06,
    epsilon_frac: float = 0.75,        # ε atteint son minimum à cette fraction de l'entraînement
    horizon: int = 120,
    seed: int = 0,
    verbose: bool = True,
    checkpoint_file: str | None = None,
    checkpoint_every: int = 1000,
    resume: bool = False,
) -> np.ndarray:
    env = TicketEnv(horizon=horizon)            # cadence de décision gérée par l'env
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n

    q_table = np.zeros((*STATE_BINS, n_actions))
    start_episode = 0

    if resume and checkpoint_file:
        ckpt = _load_checkpoint(checkpoint_file)
        if ckpt is not None:
            q_table, st = ckpt
            if q_table.shape == (*STATE_BINS, n_actions):
                start_episode = int(st.get("episode", 0))
                print(f"Reprise au checkpoint : épisode {start_episode}")
            else:
                print("Checkpoint de forme incompatible — entraînement depuis zéro.")
                q_table = np.zeros((*STATE_BINS, n_actions))

    def epsilon(ep: int) -> float:
        frac = ep / max(1.0, epsilon_frac * episodes)
        return max(epsilon_min, epsilon_start + (epsilon_min - epsilon_start) * frac)

    returns: list[float] = []
    for episode in range(start_episode, episodes):
        eps = epsilon(episode)
        obs, _ = env.reset(seed=seed + episode)     # DR reproductible par épisode
        state = discretize_observation(obs)
        traj: list[tuple[tuple, int, float]] = []
        done = False
        while not done:
            if rng.random() < eps:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(q_table[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            traj.append((state, action, reward))
            state = discretize_observation(obs)

        # --- mise à jour Monte-Carlo (retour à rebours, sans bootstrapping) --- #
        g = 0.0
        for st, ac, rw in reversed(traj):
            g = rw + gamma * g
            q_table[st][ac] += alpha * (g - q_table[st][ac])
        returns.append(float(sum(rw for _, _, rw in traj)))

        done_episodes = episode + 1
        if checkpoint_file and done_episodes % checkpoint_every == 0:
            _atomic_save(
                checkpoint_file,
                q_table,
                {"episode": done_episodes, "total": episodes, "seed": seed, "horizon": horizon},
            )
        if verbose and done_episodes % max(1, episodes // 20) == 0:
            print(
                f"  épisode {done_episodes:>6d}/{episodes}  eps={eps:.3f}  "
                f"retour_moyen={np.mean(returns[-500:]):.2f}",
                flush=True,
            )

    return q_table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60_000)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--models-dir", default=None, help="défaut : models/pricingAgents")
    p.add_argument("--resume", action="store_true", help="reprend depuis le dernier checkpoint")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    args = p.parse_args()

    ckpt = str(checkpoint_path("q_table", args.models_dir))
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    print(f"Entraînement Q-table (Monte-Carlo, {args.episodes} épisodes){' — reprise' if args.resume else ''}...")
    q_table = train(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        horizon=args.horizon,
        seed=args.seed,
        checkpoint_file=ckpt,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    final = model_path("q_table", models_dir=args.models_dir)
    np.save(str(final), q_table)
    visited = int(np.count_nonzero(q_table.any(axis=-1)))
    write_meta(
        final,
        kind="q_table",
        method="monte_carlo",
        episodes=args.episodes,
        horizon=args.horizon,
        steps_per_decision=TicketEnv(horizon=args.horizon).steps_per_decision,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        state_bins=list(STATE_BINS),
        states_visited=visited,
        states_total=int(np.prod(STATE_BINS)),
    )
    for suffix in ("", ".state.json"):             # checkpoint terminé -> nettoyage
        f = ckpt + suffix
        if os.path.exists(f):
            os.remove(f)
    print(f"Modèle sauvegardé : {final.name}  (états visités : {visited}/{int(np.prod(STATE_BINS))})")

    _verify(q_table, args.horizon)                  # contrôle de non-régression vs prix fixe


if __name__ == "__main__":
    main()
