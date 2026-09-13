"""Acheteur virtuel du simulateur de billetterie.

Chaque ``BuyerAgent`` représente un spectateur potentiel. Il ne « joue » pas à
chaque pas de temps : le modèle le sollicite uniquement lorsqu'un évènement de
visite (généré par le processus de Poisson non homogène) le concerne. À chaque
visite l'agent prend une décision d'achat via un **choix logit binaire** —
c'est-à-dire un modèle Logit Multinomial (MNL) à deux alternatives : « acheter »
(utilité ``U``) ou « ne rien faire » (utilité de référence 0).

    U      = β0 + βu·urgence − βp·prix − βb·max(0, prix/budget − BUDGET_COMFORT)
    P(achat) = exp(U) / (1 + exp(U))            (= sigmoïde de U)

* ``β0``  — intérêt intrinsèque pour l'artiste (genre + fidélité des fans)
* ``βu``  — prime d'urgence : la disposition à payer se révèle à l'approche de la
  date (``urgence`` passe de 0 au lancement à 1 la veille du concert) ; la cohorte
  « dernière minute » (peu empressée, arrivée tardive) a une prime plus forte
* ``βp``  — sensibilité au prix absolue (popularité de l'artiste + style musical)
* ``βb``  — pression budgétaire : pénalité lisse quand le prix approche / dépasse
  le budget (contrainte de solvabilité, distincte de βp qui est l'aversion au prix)
"""

from __future__ import annotations

import math

import mesa
import numpy as np

from refdata import BUDGET_COMFORT


def logistic(x):
    """Sigmoïde numériquement stable (accepte scalaire ou ``np.ndarray``)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _sigmoid_scalar(x: float) -> float:
    """Sigmoïde stable pour un scalaire Python (bien plus rapide que NumPy)."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class BuyerAgent(mesa.Agent):
    """Un spectateur potentiel doté d'une fonction d'utilité MNL."""

    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        *,
        age_bracket: str,
        budget: float,
        beta_0: float,
        beta_p: float,
        beta_urgency: float,
        beta_budget: float,
        eagerness: float,
        max_visits: int,
    ) -> None:
        super().__init__(unique_id, model)
        self.age_bracket = age_bracket
        self.budget = float(budget)
        self.beta_0 = float(beta_0)
        self.beta_p = float(beta_p)
        self.beta_urgency = float(beta_urgency)
        self.beta_budget = float(beta_budget)
        #: tendance à acheter tôt plutôt que d'attendre (0–1)
        self.eagerness = float(eagerness)
        #: nombre de visites restantes avant abandon définitif
        self.visits_left = int(max_visits)

        self.has_bought = False
        self.visits_made = 0

    # ------------------------------------------------------------------ #

    @property
    def is_active(self) -> bool:
        """L'agent peut-il encore visiter la plateforme ?"""
        return not self.has_bought and self.visits_left > 0

    def purchase_probability(self, price: float, urgency: float) -> float:
        """Probabilité d'achat pour un prix et un niveau d'urgence donnés."""
        utility = self.beta_0 + self.beta_urgency * urgency - self.beta_p * price
        overrun = price / self.budget - BUDGET_COMFORT
        if overrun > 0.0:
            utility -= self.beta_budget * overrun          # pression budgétaire (lisse)
        return _sigmoid_scalar(utility)

    def visit(self) -> bool:
        """Une visite de la plateforme : renvoie ``True`` si un billet est acheté."""
        if self.has_bought or self.visits_left <= 0:
            return False

        self.visits_made += 1
        self.visits_left -= 1

        model = self.model
        prob = self.purchase_probability(model.current_price, model.urgency)
        if model.rng.random() < prob and model.attempt_purchase():
            self.has_bought = True
            return True
        return False

    # ``step`` reste défini pour compatibilité avec l'API Mesa, mais le modèle
    # pilote les visites explicitement via ``visit()``.
    def step(self) -> None:  # pragma: no cover - non utilisé
        self.visit()
