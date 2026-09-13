"""Briques du DQN « from scratch » (TensorFlow / Keras).

Ces éléments sont volontairement minimalistes et partagés entre le script
``scripts/train_dqn.py`` et le notebook 08 : le réseau de neurones, le tampon
de rejeu (replay buffer) et un pas de descente de gradient DQN. La boucle
d'entraînement elle-même reste visible dans les deux (comme pour la Q-table).
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence les logs d'init TF

import numpy as np
import tensorflow as tf


def build_qnetwork(obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)) -> tf.keras.Model:
    """Réseau Q : observation -> une Q-value par action (couche de sortie linéaire)."""
    model = tf.keras.Sequential(name="qnet")
    model.add(tf.keras.layers.Input(shape=(int(obs_dim),)))
    for units in hidden:
        model.add(tf.keras.layers.Dense(int(units), activation="relu"))
    model.add(tf.keras.layers.Dense(int(n_actions), activation="linear"))
    return model


class ReplayBuffer:
    """Tampon circulaire de transitions (s, a, r, s', terminé)."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.size = 0
        self._ptr = 0

    def add(self, s, a, r, s2, done) -> None:
        i = self._ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, float(done)
        self._ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx])

    # -- persistance (pour la reprise) --------------------------------- #

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, s=self.s, a=self.a, r=self.r, s2=self.s2, done=self.done,
            size=self.size, ptr=self._ptr,
        )

    def load(self, path: str) -> None:
        d = np.load(path)
        self.s, self.a, self.r, self.s2, self.done = d["s"], d["a"], d["r"], d["s2"], d["done"]
        self.size, self._ptr = int(d["size"]), int(d["ptr"])
        self.capacity = self.s.shape[0]


def make_train_step(online: tf.keras.Model, target: tf.keras.Model, optimizer, gamma: float):
    """Renvoie une fonction ``train_step(batch) -> loss`` compilée (``tf.function``).

    Cible DQN : y = r + gamma * (1 - terminé) * max_a' Q_cible(s', a')
    Perte : Huber entre y et Q_online(s, a).
    """
    n_actions = online.output_shape[-1]

    @tf.function(reduce_retracing=True)
    def train_step(s, a, r, s2, done):
        q_next = tf.reduce_max(target(s2, training=False), axis=1)
        y = r + gamma * (1.0 - done) * q_next
        with tf.GradientTape() as tape:
            q = online(s, training=True)
            q_a = tf.reduce_sum(q * tf.one_hot(a, n_actions), axis=1)
            loss = tf.reduce_mean(tf.keras.losses.huber(y, q_a))
        grads = tape.gradient(loss, online.trainable_variables)
        optimizer.apply_gradients(zip(grads, online.trainable_variables))
        return loss

    return train_step


def greedy_action(model: tf.keras.Model, obs: np.ndarray) -> int:
    q = model(np.asarray(obs, np.float32)[None, :], training=False)
    return int(tf.argmax(q[0]).numpy())
