"""Interface Gymnasium autour du simulateur Mesa.

Formalise le Processus de Décision Markovien :

* **État** (``Box`` normalisée dans [0, 1]) :
  ``[temps_restant, inventaire_restant, popularité, rythme_de_vente,
     prix_vs_référence, fan_cost_index, dernier_prix]``
  - ``prix_vs_référence`` = position du prix courant vs le prix de référence à
    mémoire lente du marché (``0.5`` = au niveau, ``> 0.5`` = rabais). Sans lui
    l'intensité de trafic endogène dépendrait d'un état caché (``p_ref``) et le
    MDP ne serait plus markovien.
  - ``fan_cost_index`` = tier de la salle (revenu annexe potentiel par spectateur) :
    l'agent peut ainsi tarifer différemment un petit club et un stade.
* **Action** (``Discrete``) : un palier de prix parmi ``num_price_levels``
* **Récompense** : **marge de contribution** du pas, normalisée :

      (prix − marginal_cost + ancillary_capture · Fan Cost Index) · ventes

  - ``marginal_cost`` = coût variable par spectateur (frais de billetterie,
    personnel / sécurité / nettoyage au prorata, traitement du paiement). Donne
    un **prix plancher économique** : sans lui, brader au minimum reste rentable
    grâce au terme annexe et l'agent apprend une course vers le bas.
  - ``ancillary_capture`` = part des spectateurs générant un revenu annexe
    (merch, boissons, parking) — Coates & Humphreys 2007 ; Krautmann & Berri 2007.
  Les deux sont des paramètres *modélisés*, pas des curseurs de calage.
  ``info["revenue_total"]`` utilise **la même formule** -> le KPI reporté est
  exactement l'objectif optimisé. ``info`` expose aussi le détail :
  ``ticket_revenue_total`` (billetterie brute), ``ancillary_revenue_total``,
  ``variable_cost_total``.

  La récompense est **normalisée par épisode** : divisée par ``≈ 1.5 · demande
  convertible`` (min entre la capacité et 75 % des acheteurs potentiels). Un
  concert de club et un stade produisent alors des retours de magnitude
  comparable -> le bruit inter-scénarios de la randomisation de domaine (facteur
  ~40 sur le CA brut) chute d'un ordre de grandeur et l'apprentissage RL cesse
  d'être dominé par « quelle taille de salle ai-je tirée ».

Bonus terminal optionnel sur le taux de remplissage (``fill_bonus_coef``).

À chaque ``reset()`` un nouvel épisode de randomisation de domaine démarre :
artiste et salle tirés au sort, demande bruitée, population régénérée.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from market import TicketMarketModel

#: Bornes de discrétisation pour l'agent Q-Learning tabulaire, sur les 6
#: premières dimensions de l'observation :
#: (temps, inventaire, popularité, rythme de vente, prix vs référence, Fan Cost Index).
#: Grille volontairement grossière (6 480 états) : le tabulaire doit voir chaque
#: état des dizaines de fois pour converger, et la randomisation de domaine
#: disperse déjà énormément les épisodes.
STATE_BINS: tuple[int, int, int, int, int, int] = (6, 6, 5, 4, 3, 3)


def discretize_observation(obs: np.ndarray, bins: tuple[int, ...] = STATE_BINS) -> tuple[int, ...]:
    """Projette l'observation continue sur la grille discrète de la Q-table."""
    return tuple(
        min(b - 1, max(0, int(float(obs[i]) * b))) for i, b in enumerate(bins)
    )


class TicketEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        min_price: float = 30.0,
        max_price: float = 120.0,
        num_price_levels: int = 10,
        horizon: int = 120,
        steps_per_decision: int = 10,
        domain_randomize: bool = True,
        reward_scale: float | str = "auto",
        ancillary_capture: float = 0.35,
        marginal_cost: float = 15.0,
        fill_bonus_coef: float = 0.0,
        data_dir: str | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if min_price <= 0 or max_price <= min_price:
            raise ValueError("Il faut 0 < min_price < max_price.")
        if num_price_levels < 1:
            raise ValueError("num_price_levels doit valoir au moins 1.")

        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.num_price_levels = int(num_price_levels)
        self.horizon = int(horizon)
        #: pas de marché entre deux révisions de prix (un organisateur révise le
        #: tarif périodiquement, pas en continu). Ramène l'horizon de décision de
        #: ``horizon`` à ``horizon / steps_per_decision`` -> attribution de crédit
        #: temporel tractable pour le Q-Learning / DQN.
        self.steps_per_decision = max(1, int(steps_per_decision))
        self.domain_randomize = bool(domain_randomize)
        self.reward_scale_cfg = reward_scale
        #: part des spectateurs générant un revenu annexe (merch, boissons, parking)
        #: — Coates & Humphreys 2007 ; Krautmann & Berri 2007. Pondère le FCI dans
        #: la récompense ET dans info["revenue_total"] (KPI reporté == objectif).
        self.ancillary_capture = float(ancillary_capture)
        #: coût variable par spectateur (CHF) : billetterie + personnel/sécurité/
        #: nettoyage au prorata + traitement du paiement. Prix plancher économique.
        self.marginal_cost = float(marginal_cost)
        self.fill_bonus_coef = float(fill_bonus_coef)
        self.data_dir = data_dir
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.num_price_levels)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.market: TicketMarketModel | None = None
        self._last_price = 0.5 * (self.min_price + self.max_price)
        self._reward_scale = 1.0
        self._revenue_total = 0.0
        self._ticket_revenue_total = 0.0
        self._ancillary_revenue_total = 0.0
        self._variable_cost_total = 0.0

    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        self.market = TicketMarketModel(
            artist=options.get("artist"),
            venue=options.get("venue"),
            horizon=int(options.get("horizon", self.horizon)),
            initial_tickets=options.get("initial_tickets"),
            demand_scale=float(options.get("demand_scale", 1.0)),
            domain_randomize=bool(options.get("domain_randomize", self.domain_randomize)),
            rng=self.np_random,
            data_dir=self.data_dir,
        )

        self._last_price = 0.5 * (self.min_price + self.max_price)
        self._revenue_total = 0.0
        self._ticket_revenue_total = 0.0
        self._ancillary_revenue_total = 0.0
        self._variable_cost_total = 0.0
        if self.reward_scale_cfg == "auto":
            # Normalisation par épisode : ~ prix net de référence (58 CHF) / retour
            # cible (~38) x demande convertible du scénario. Rend club et stade
            # comparables (cf. docstring du module).
            convertible = min(
                self.market.initial_tickets,
                0.75 * self.market.n_potential_buyers,
            )
            self._reward_scale = max(1.0, 1.5 * convertible)
        else:
            self._reward_scale = float(self.reward_scale_cfg)

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if self.market is None:
            raise RuntimeError("Appeler reset() avant step().")

        price = self._decode_action(int(action))
        self._last_price = price

        # Le prix est maintenu pendant ``steps_per_decision`` pas de marché ; la
        # récompense de la décision est la marge de contribution agrégée sur le bloc.
        block_contribution = 0.0
        for _ in range(self.steps_per_decision):
            if self.market.done:
                break
            self.market.current_price = price
            self.market.step()

            sold = self.market.tickets_sold_this_step
            ticket_rev = price * sold
            ancillary_rev = self.ancillary_capture * self.market.fan_cost_index * sold
            variable_cost = self.marginal_cost * sold
            contribution = ticket_rev + ancillary_rev - variable_cost

            self._ticket_revenue_total += ticket_rev
            self._ancillary_revenue_total += ancillary_rev
            self._variable_cost_total += variable_cost
            self._revenue_total += contribution
            block_contribution += contribution

        reward = block_contribution / self._reward_scale

        terminated = self.market.done
        truncated = False
        if terminated and self.fill_bonus_coef:
            reward += self.fill_bonus_coef * self.market.fill_rate

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ #

    def _decode_action(self, action: int) -> float:
        if self.num_price_levels == 1:
            return self.min_price
        step_size = (self.max_price - self.min_price) / (self.num_price_levels - 1)
        return round(self.min_price + action * step_size, 2)

    def _get_obs(self) -> np.ndarray:
        m = self.market
        pace = float(np.clip(m.sales_pace_ratio(self.steps_per_decision) / 2.0, 0.0, 1.0))
        last_price_norm = (self._last_price - self.min_price) / (self.max_price - self.min_price)
        # Position du prix courant vs prix de référence à mémoire lente du marché
        # (échelle log symétrique : ratio 1 -> 0.5, rabais de moitié -> 1, premium ×2 -> 0).
        ref = m.reference_price or self._last_price
        ratio = min(4.0, max(0.25, ref / max(self._last_price, 1e-6)))
        price_vs_ref = 0.5 + 0.5 * math.log(ratio) / math.log(2.0)
        fci_norm = (m.fan_cost_index - 25.0) / 25.0        # salles réelles : FCI 25..50 CHF
        return np.array(
            [
                m.time_ratio,
                m.inventory_ratio,
                (m.popularity_index - 1.0) / 9.0,
                pace,
                float(np.clip(price_vs_ref, 0.0, 1.0)),
                float(np.clip(fci_norm, 0.0, 1.0)),
                float(np.clip(last_price_norm, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        m = self.market
        return {
            "tickets_sold": m.tickets_sold_this_step,
            "tickets_sold_total": m.tickets_sold_total,
            "revenue_total": self._revenue_total,               # marge de contribution (== objectif RL)
            "ca_total": self._ticket_revenue_total + self._ancillary_revenue_total,  # CA = billetterie + revenus annexes
            "ticket_revenue_total": self._ticket_revenue_total,  # billetterie brute (prix x ventes)
            "ancillary_revenue_total": self._ancillary_revenue_total,
            "variable_cost_total": self._variable_cost_total,
            "fill_rate": m.fill_rate,
            "price": self._last_price,
            "popularity_index": m.popularity_index,
            "artist": m.artist.name,
            "venue": m.venue.name,
            "time_ratio": m.time_ratio,
            "inventory_ratio": m.inventory_ratio,
        }

    def render(self):
        if self.render_mode != "human" or self.market is None:
            return
        m = self.market
        print(
            f"[{m.artist.name} @ {m.venue.name}] t={m.time_remaining:3d} "
            f"stock={m.tickets_remaining:5d}/{m.initial_tickets} "
            f"prix={self._last_price:6.2f} vendus_total={m.tickets_sold_total}"
        )
