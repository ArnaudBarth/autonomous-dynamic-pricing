"""Agent basé sur une Q-table (valeurs état-action estimées par contrôle
Monte-Carlo, cf. ``scripts/train_qtable.py`` / notebook 07).

La discrétisation de l'état est importée depuis ``gym_wrapper`` : l'agent
d'inférence et l'entraînement utilisent donc *exactement* la même grille, ce qui
évite d'appliquer une politique apprise sur un état mal aligné.
"""

from __future__ import annotations

import numpy as np

try:  # ../env et ../pricing_agent sur le sys.path (cas des notebooks / scripts)
    from gym_wrapper import STATE_BINS, discretize_observation
    from registry import read_meta, resolve_model
except ImportError:  # pragma: no cover - exécution en package
    from env.gym_wrapper import STATE_BINS, discretize_observation
    from pricing_agent.registry import read_meta, resolve_model


class QTableAgent:
    name = "qtable"

    def __init__(self, q_table: np.ndarray, epsilon: float = 0.0):
        self.q_table = np.asarray(q_table)
        self.epsilon = float(epsilon)
        expected_ndim = len(STATE_BINS) + 1
        if self.q_table.ndim != expected_ndim:
            raise ValueError(
                f"Q-table de dimension {self.q_table.ndim}, attendu {expected_ndim} "
                f"(bins {STATE_BINS} + actions)."
            )

    @classmethod
    def load(cls, version: str | None = None, epsilon: float = 0.0, models_dir=None) -> "QTableAgent":
        """Charge une Q-table. ``version=None`` -> la plus récente ; sinon un
        horodatage (fragment), ou un chemin de fichier complet."""
        path = resolve_model("q_table", version, models_dir)
        agent = cls(np.load(path), epsilon=epsilon)
        agent.path = path
        agent.meta = read_meta(path)
        return agent

    def predict(self, obs, deterministic: bool = True):
        if not deterministic and self.epsilon > 0.0 and np.random.random() < self.epsilon:
            return int(np.random.randint(self.q_table.shape[-1])), None
        state = discretize_observation(np.asarray(obs))
        return int(np.argmax(self.q_table[state])), None
