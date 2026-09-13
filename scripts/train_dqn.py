"""Entraînement d'un agent Deep Q-Network (TensorFlow, « from scratch »).

Réseau MLP Keras + replay buffer + réseau cible + ε-greedy + équation de Bellman
DQN. Interface identique au Q-Learning tabulaire (même environnement, même
récompense) — c'est la phase 2 du projet : passer de la table discrète à un
approximateur de fonction pour l'espace d'états continu.

    python scripts/train_dqn.py --episodes 20000
    python scripts/train_dqn.py --episodes 20000 --resume    # reprend après une coupure

Checkpoint (modèle en ligne + cible + replay buffer + état) tous les
``--checkpoint-every`` épisodes, écriture atomique. ``--resume`` recharge tout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing_agent"))

import tensorflow as tf

from gym_wrapper import TicketEnv
from dqn import ReplayBuffer, build_qnetwork, make_train_step
from registry import checkpoint_path, model_path, write_meta


def _epsilon(step: int, total_steps: int, start: float = 1.0, end: float = 0.05, frac: float = 0.7) -> float:
    """Décroissance linéaire de ``start`` à ``end`` sur ``frac`` de l'entraînement."""
    if step >= frac * total_steps:
        return end
    return start + (end - start) * (step / (frac * total_steps))


def _verify(online, horizon: int, n_episodes: int = 150) -> bool:
    """Contrôle de non-régression : la politique gloutonne bat-elle le prix fixe
    sur le CA ? Un réseau entraîné sur un environnement périmé passe la validation
    de forme mais échoue ici. Retourne True si le CA moyen dépasse le prix fixe 70.
    """
    from agent_fixed import FixedPriceAgent
    from agent_dqn import DQNAgent

    fixed = FixedPriceAgent.from_price(70.0)
    learned = DQNAgent(online)
    ca_fixed, ca_learned = 0.0, 0.0
    for i in range(n_episodes):
        for agent, is_fixed in ((fixed, True), (learned, False)):
            env = TicketEnv(horizon=horizon)
            obs, _ = env.reset(seed=100_000 + i)
            done = False
            info: dict = {}
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(action)
                done = term or trunc
            if is_fixed:
                ca_fixed += info["ca_total"]
            else:
                ca_learned += info["ca_total"]
    gain = ca_learned / ca_fixed - 1.0
    print(f"Verification ({n_episodes} scenarios apparies) : CA vs prix fixe 70 = {gain:+.1%}")
    if gain <= 0:
        print("  [ATTENTION] le modele ne bat PAS le prix fixe -- probable environnement perime.\n"
              "  Relance dans un terminal neuf :  python scripts/train_dqn.py --episodes 15000")
    return gain > 0


def train(
    episodes: int = 20_000,
    horizon: int = 120,
    gamma: float = 0.97,
    lr: float = 5e-4,
    batch_size: int = 64,
    buffer_capacity: int = 200_000,
    warmup_steps: int = 1_500,
    train_every: int = 1,               # ~12 décisions par épisode (cadence grossière)
    target_update_every: int = 1_000,
    hidden: tuple[int, ...] = (128, 128),
    seed: int = 0,
    checkpoint_prefix: str | None = None,
    checkpoint_every: int = 250,
    resume: bool = False,
    verbose: bool = True,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    env = TicketEnv(horizon=horizon)
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    rng = np.random.default_rng(seed)

    online = build_qnetwork(obs_dim, n_actions, hidden)
    target = build_qnetwork(obs_dim, n_actions, hidden)
    target.set_weights(online.get_weights())
    optimizer = tf.keras.optimizers.Adam(lr)
    train_step = make_train_step(online, target, optimizer, gamma)
    buffer = ReplayBuffer(buffer_capacity, obs_dim)

    decisions_per_ep = max(1, horizon // env.steps_per_decision)
    total_steps_est = episodes * decisions_per_ep
    start_episode, global_step = 0, 0
    returns: list[float] = []

    ck = _checkpoint_paths(checkpoint_prefix) if checkpoint_prefix else None
    if resume and ck and os.path.exists(ck["state"]):
        st = json.load(open(ck["state"], encoding="utf-8"))
        online.load_weights(ck["online"])
        target.load_weights(ck["target"])
        buffer.load(ck["buffer"])
        start_episode, global_step = st["episode"], st["global_step"]
        returns = st.get("returns", [])
        print(f"Reprise : épisode {start_episode}, pas {global_step}", flush=True)

    @tf.function(reduce_retracing=True)
    def q_online(s):
        return online(s, training=False)

    t0 = time.time()
    for episode in range(start_episode, episodes):
        obs, _ = env.reset(seed=seed + episode)          # DR reproductible par épisode
        done = False
        ep_return = 0.0

        while not done:
            eps = _epsilon(global_step, total_steps_est)
            if rng.random() < eps:
                action = int(rng.integers(n_actions))
            else:
                action = int(tf.argmax(q_online(obs[None, :])[0]).numpy())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, terminated)  # bootstrap seulement si terminé
            obs = next_obs
            ep_return += reward
            global_step += 1

            if buffer.size >= warmup_steps and global_step % train_every == 0:
                train_step(*(tf.convert_to_tensor(x) for x in buffer.sample(batch_size, rng)))
            if global_step % target_update_every == 0:
                target.set_weights(online.get_weights())

        returns.append(ep_return)
        done_ep = episode + 1

        if ck and done_ep % checkpoint_every == 0:
            _save_checkpoint(ck, online, target, buffer,
                             {"episode": done_ep, "global_step": global_step, "returns": returns})
        if verbose and done_ep % max(1, episodes // 25) == 0:
            rate = (done_ep - start_episode) / (time.time() - t0)
            print(
                f"  épisode {done_ep:>5d}/{episodes}  eps={_epsilon(global_step, total_steps_est):.3f}  "
                f"retour_moyen={np.mean(returns[-100:]):.2f}  ({rate:.1f} ép/s)",
                flush=True,
            )

    return online


# --------------------------------------------------------------------------- #
# Checkpoints (écriture atomique)                                              #
# --------------------------------------------------------------------------- #


def _checkpoint_paths(prefix: str) -> dict[str, str]:
    return {
        "online": prefix + ".online.weights.h5",
        "target": prefix + ".target.weights.h5",
        "buffer": prefix + ".buffer.npz",
        "state": prefix + ".state.json",
    }


def _cleanup_checkpoint(prefix: str) -> None:
    for f in _checkpoint_paths(prefix).values():
        if os.path.exists(f):
            os.remove(f)


def _save_checkpoint(ck: dict, online, target, buffer: ReplayBuffer, state: dict) -> None:
    online.save_weights(ck["online"])
    target.save_weights(ck["target"])
    buffer.save(ck["buffer"])
    tmp = ck["state"] + ".tmp"
    json.dump(state, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, ck["state"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--horizon", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.97)   # identique au tabulaire (même MDP)
    p.add_argument("--models-dir", default=None, help="défaut : models/pricingAgents")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=250)
    args = p.parse_args()

    ckpt = str(checkpoint_path("dqn", args.models_dir))
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    print(f"Entraînement DQN TensorFlow ({args.episodes} épisodes){' — reprise' if args.resume else ''}...")
    online = train(
        episodes=args.episodes, horizon=args.horizon, seed=args.seed,
        lr=args.lr, gamma=args.gamma,
        checkpoint_prefix=ckpt, checkpoint_every=args.checkpoint_every, resume=args.resume,
    )

    final = model_path("dqn", models_dir=args.models_dir)
    online.save(str(final))
    write_meta(
        final,
        kind="dqn",
        episodes=args.episodes,
        horizon=args.horizon,
        steps_per_decision=TicketEnv(horizon=args.horizon).steps_per_decision,
        seed=args.seed,
        gamma=args.gamma,
        lr=args.lr,
        architecture=[layer.units for layer in online.layers if hasattr(layer, "units")],
    )
    _cleanup_checkpoint(ckpt)
    print(f"Modèle sauvegardé : {final.name}")

    _verify(online, args.horizon)                   # contrôle de non-régression vs prix fixe


if __name__ == "__main__":
    main()
