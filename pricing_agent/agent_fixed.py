"""Baseline : tarification statique (un prix unique pour toute la vente)."""

from __future__ import annotations

from base import price_to_action


class FixedPriceAgent:
    """Propose toujours le même palier de prix (approche historique de l'industrie)."""

    name = "fixed"

    def __init__(self, action: int = 4):
        self.action = int(action)

    @classmethod
    def from_price(
        cls,
        price: float,
        min_price: float = 30.0,
        max_price: float = 120.0,
        num_price_levels: int = 10,
    ) -> "FixedPriceAgent":
        return cls(price_to_action(price, min_price, max_price, num_price_levels))

    def predict(self, obs, deterministic: bool = True):
        return self.action, None
