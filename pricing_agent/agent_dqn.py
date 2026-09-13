"""Agent Deep Q-Network (TensorFlow) — chargement + inférence.

L'entraînement se fait dans ``scripts/train_dqn.py`` / le notebook 08 ; ce module
ne sert qu'à rejouer la politique apprise pour la comparaison avec les baselines.
Même interface ``predict(obs, deterministic=True)`` que les autres agents.
"""

from __future__ import annotations

import numpy as np

try:
    from registry import read_meta, resolve_model
except ImportError:  # pragma: no cover
    from pricing_agent.registry import read_meta, resolve_model


class DQNAgent:
    name = "dqn"

    def __init__(self, model, epsilon: float = 0.0):
        self.model = model
        self.epsilon = float(epsilon)
        self._n_actions = int(model.output_shape[-1])

    @classmethod
    def load(cls, version: str | None = None, epsilon: float = 0.0, models_dir=None) -> "DQNAgent":
        """Charge un modèle DQN. ``version=None`` -> le plus récent ; sinon un
        horodatage (fragment), ou un chemin de fichier complet."""
        import tensorflow as tf  # import paresseux : TF est lourd

        path = resolve_model("dqn", version, models_dir)
        agent = cls(tf.keras.models.load_model(path), epsilon=epsilon)
        agent.path = path
        agent.meta = read_meta(path)
        return agent

    def q_values(self, obs) -> np.ndarray:
        x = np.asarray(obs, np.float32)[None, :]
        return np.asarray(self.model(x, training=False))[0]

    def predict(self, obs, deterministic: bool = True):
        if not deterministic and self.epsilon > 0.0 and np.random.random() < self.epsilon:
            return int(np.random.randint(self._n_actions)), None
        return int(np.argmax(self.q_values(obs))), None
