"""Interface commune des agents de tarification."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PricingAgent(Protocol):
    """Contrat minimal, aligné sur ``stable_baselines3.BaseAlgorithm.predict``."""

    name: str

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, Any]:
        """Renvoie ``(action, state)`` où ``action`` est un indice de palier de prix."""
        ...


def price_to_action(price: float, min_price: float, max_price: float, num_price_levels: int) -> int:
    """Indice de palier le plus proche d'un prix cible (CHF)."""
    if num_price_levels <= 1:
        return 0
    step = (max_price - min_price) / (num_price_levels - 1)
    return int(np.clip(round((price - min_price) / step), 0, num_price_levels - 1))


def action_to_price(action: int, min_price: float, max_price: float, num_price_levels: int) -> float:
    if num_price_levels <= 1:
        return min_price
    step = (max_price - min_price) / (num_price_levels - 1)
    return round(min_price + action * step, 2)
