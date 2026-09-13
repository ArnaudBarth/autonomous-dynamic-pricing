"""Simulateur de marché de billetterie basé sur les agents (Mesa).

Le modèle représente une salle à capacité fixe, un compte à rebours jusqu'à la
date du concert, et une population d'acheteurs virtuels hétérogènes. À chaque
pas de temps :

1. un processus de Poisson non homogène génère des **visites** de la plateforme.
   Son intensité varie dans le temps (pic à l'ouverture, plancher au milieu,
   accélération avant la date) et **réagit au prix** relativement à la *courbe de
   prix attendue* par le public (:meth:`expected_price`) : tarifer en dessous
   attire les chasseurs de bonnes affaires, au-dessus raréfie le trafic ; un
   stock qui fond près de la date déclenche une ruée d'achat ;
2. chaque agent visité prend une décision d'achat via son modèle Logit (MNL) ;
3. le stock et le temps restant sont mis à jour.

**Randomisation de domaine** — à chaque construction du modèle (donc à chaque
épisode d'entraînement) : l'artiste et la salle sont tirés au sort dans les
données ChartMetric / OFS, la demande potentielle est bruitée, et chaque agent
reçoit un budget, un intérêt et une sensibilité au prix propres. L'agent de
pricing ne peut donc pas « apprendre par cœur » un scénario figé.
"""

from __future__ import annotations

import mesa
import numpy as np

from agents import BuyerAgent, logistic
from refdata import (
    AGE_BRACKETS,
    BUDGET_PRESSURE,
    BUDGET_PRESSURE_NOISE_SD,
    DEMAND_MAX_RATIO,
    DEMAND_MIN_BUYERS,
    DEMAND_NOISE_SD,
    DEMAND_RATIO_BASE,
    DEMAND_RATIO_POP_GAIN,
    INTEREST_BASE,
    INTEREST_GENRE_GAIN,
    INTEREST_LOYALTY_GAIN,
    INTEREST_NOISE_SD,
    LATE_INTEREST_BONUS,
    LATE_PRICE_SENS_FACTOR,
    LATE_URGENCY_BONUS,
    PRICE_ANCHOR_BASE,
    PRICE_ANCHOR_NOISE_SD,
    PRICE_ANCHOR_POP_SLOPE,
    PRICE_ESCALATION_EXPECTED,
    PRICE_SENS_BASE,
    PRICE_SENS_MAX,
    PRICE_SENS_MIN,
    PRICE_SENS_NOISE_SD,
    PRICE_SENS_POP_DROP,
    PRICE_SENS_STYLE_DROP,
    TRAFFIC_MULT_MAX,
    TRAFFIC_MULT_MIN,
    TRAFFIC_PRICE_ELASTICITY,
    TRAFFIC_SCARCITY_GAIN,
    URGENCY_BASE,
    URGENCY_NOISE_SD,
    URGENCY_PATIENCE_GAIN,
    VISIT_EAGER_LAUNCH_GAIN,
    VISIT_EVENT_TAU,
    VISIT_FLOOR,
    VISIT_LATE_EVENT_GAIN,
    VISIT_LATE_FRACTION,
    VISIT_LAUNCH_TAU,
    VISITS_BASE,
    VISITS_STYLE_GAIN,
    ArtistProfile,
    DemographicProfile,
    GenreProfile,
    VenueProfile,
)


class TicketMarketModel(mesa.Model):
    """Environnement stochastique de vente de billets pour un concert."""

    def __init__(
        self,
        *,
        artist: ArtistProfile | str | None = None,
        venue: VenueProfile | str | None = None,
        horizon: int = 120,
        initial_tickets: int | None = None,
        demand_scale: float = 1.0,
        domain_randomize: bool = True,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        data_dir: str | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(seed)

        # -- scénario (randomisation de domaine) ------------------------- #
        if isinstance(artist, str):
            artist = ArtistProfile.by_name(artist, data_dir)
        if isinstance(venue, str):
            venue = VenueProfile.by_name(venue, data_dir)
        self.artist: ArtistProfile = artist or ArtistProfile.sample(self.rng, data_dir)
        if venue is not None:
            self.venue: VenueProfile = venue
        elif domain_randomize:
            # Salle cohérente avec la popularité (évite artiste de niche / stade).
            self.venue = VenueProfile.sample_for_popularity(
                self.rng, self.artist.popularity_index, data_dir
            )
        else:
            self.venue = VenueProfile.sample(self.rng, data_dir)
        self.genre: GenreProfile = GenreProfile.lookup(self.artist.genre, data_dir)
        self._demo: DemographicProfile = DemographicProfile.from_csv(data_dir)

        self.popularity_index: float = self.artist.popularity_index
        self.fan_cost_index: float = self.venue.fan_cost_index

        # Ancre de la courbe de prix attendu (superstar -> plus cher d'emblée).
        anchor = PRICE_ANCHOR_BASE + PRICE_ANCHOR_POP_SLOPE * (self.popularity_index - 1.0)
        if domain_randomize:
            anchor *= float(np.exp(self.rng.normal(0.0, PRICE_ANCHOR_NOISE_SD)))
        self._price_anchor: float = float(anchor)

        # -- capacité / horizon --------------------------------------- #
        self.initial_tickets: int = int(initial_tickets or self.venue.capacity)
        self.tickets_remaining: int = self.initial_tickets
        self.initial_time: int = int(horizon)
        self.time_remaining: int = int(horizon)

        # -- état de vente ------------------------------------------- #
        self.current_price: float | None = None
        self.tickets_sold_this_step: int = 0
        self.tickets_sold_total: int = 0
        self.sales_history: list[int] = []
        self._step_index: int = 0
        #: urgence temporelle courante (0 au lancement -> 1 la veille du concert)
        self.urgency: float = 0.0

        # -- population d'acheteurs --------------------------------- #
        self._build_population(demand_scale, domain_randomize)
        self._build_visit_intensity()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "time_remaining": "time_remaining",
                "tickets_remaining": "tickets_remaining",
                "tickets_sold_step": "tickets_sold_this_step",
                "tickets_sold_total": "tickets_sold_total",
                "price": "current_price",
            }
        )

    # ------------------------------------------------------------------ #
    # Construction de la population                                       #
    # ------------------------------------------------------------------ #

    def _build_population(self, demand_scale: float, domain_randomize: bool) -> None:
        rng = self.rng
        pop = self.popularity_index
        mobility = self.genre.mobility_factor

        demand_ratio = DEMAND_RATIO_BASE + DEMAND_RATIO_POP_GAIN * pop
        noise = float(np.exp(rng.normal(0.0, DEMAND_NOISE_SD))) if domain_randomize else 1.0
        noise = float(np.clip(noise, 0.4, 2.3))          # borne le bruit log-normal
        raw = self.initial_tickets * demand_ratio * noise * demand_scale
        # Plafond à DEMAND_MAX_RATIO × capacité : garde les épisodes rapides et évite
        # les superstars « impossibles » (cf. éval du simulateur).
        n_buyers = int(np.clip(raw, DEMAND_MIN_BUYERS, DEMAND_MAX_RATIO * self.initial_tickets))
        self.n_potential_buyers = n_buyers

        # Âge : distribution ChartMetric pondérée par la participation OFS.
        weights = self.artist.effective_age_weights(self._demo)
        brackets = np.array(AGE_BRACKETS)
        probs = np.array([weights[b] for b in AGE_BRACKETS])
        probs = probs / probs.sum()  # garantit une somme exacte à 1 pour rng.choice
        age_idx = rng.choice(len(brackets), size=n_buyers, p=probs)
        agent_brackets = brackets[age_idx]

        # Budget : salaire médian de la tranche * part + bruit triangulaire.
        budgets = self._demo.sample_budgets(agent_brackets, rng)

        # Intérêt (bêta_0) : genre + fidélité des fans + bruit individuel.
        beta_0 = (
            INTEREST_BASE
            + INTEREST_GENRE_GAIN * self.genre.base_conversion_rate
            + INTEREST_LOYALTY_GAIN * self.artist.engagement_ratio
            + rng.normal(0.0, INTEREST_NOISE_SD, size=n_buyers)
        )

        # Sensibilité au prix (bêta_p) : baisse avec la popularité et le style.
        beta_p = (
            PRICE_SENS_BASE
            - PRICE_SENS_POP_DROP * (pop - 1.0)
            - PRICE_SENS_STYLE_DROP * (mobility - 1.0)
            + rng.normal(0.0, PRICE_SENS_NOISE_SD, size=n_buyers)
        )
        beta_p = np.clip(beta_p, PRICE_SENS_MIN, PRICE_SENS_MAX)

        # Empressement (achat précoce vs attente) dérivé de l'intérêt.
        eagerness = np.clip(logistic(beta_0 - 0.3) + rng.normal(0.0, 0.05, n_buyers), 0.05, 0.95)

        # Cohorte « dernière minute » : tirée de préférence chez les peu empressés.
        p_late = np.clip(VISIT_LATE_FRACTION * 2.0 * (1.0 - eagerness), 0.0, 1.0)
        is_late = rng.random(n_buyers) < p_late
        self._is_late = is_late

        # Prime d'urgence : les agents patients révèlent une DAP plus forte tard,
        # la cohorte « dernière minute » encore davantage.
        beta_urgency = np.clip(
            URGENCY_BASE
            + URGENCY_PATIENCE_GAIN * (1.0 - eagerness)
            + LATE_URGENCY_BONUS * is_late
            + rng.normal(0.0, URGENCY_NOISE_SD, n_buyers),
            0.0,
            None,
        )
        # Une fois décidés, les acheteurs tardifs sont plus intéressés et moins
        # regardants sur le prix — c'est le segment que le vendeur skimme tard.
        beta_0 = beta_0 + LATE_INTEREST_BONUS * is_late
        beta_p = np.where(is_late, beta_p * LATE_PRICE_SENS_FACTOR, beta_p)

        # Pression budgétaire individuelle (certains sont stricts, d'autres flexibles).
        beta_budget = np.clip(
            BUDGET_PRESSURE + rng.normal(0.0, BUDGET_PRESSURE_NOISE_SD, n_buyers), 1.0, None
        )

        # Nombre de visites avant abandon (mobilité du style -> plus de visites).
        mean_visits = VISITS_BASE + VISITS_STYLE_GAIN * (mobility - 1.0)
        max_visits = np.maximum(1, rng.poisson(mean_visits, size=n_buyers))

        self.buyers: list[BuyerAgent] = []
        for i in range(n_buyers):
            agent = BuyerAgent(
                i,
                self,
                age_bracket=str(agent_brackets[i]),
                budget=float(budgets[i]),
                beta_0=float(beta_0[i]),
                beta_p=float(beta_p[i]),
                beta_urgency=float(beta_urgency[i]),
                beta_budget=float(beta_budget[i]),
                eagerness=float(eagerness[i]),
                max_visits=int(max_visits[i]),
            )
            self.buyers.append(agent)

        self._eagerness = eagerness
        self._max_visits = max_visits

    def _build_visit_intensity(self) -> None:
        """Prépare l'intensité de visite de chaque agent au fil du temps.

        L'intensité individuelle ``λ_i(t)`` est la somme de trois profils
        temporels normalisés — pic à l'ouverture, plancher constant, rappel
        avant la date — pondérés par l'empressement de l'agent et par son
        appartenance éventuelle à la cohorte « dernière minute ». Sa somme sur
        l'horizon vaut ``max_visits_i`` (le budget de visites de l'agent).

        La superposition sur la foule donne un **processus de Poisson non
        homogène** ; à chaque pas son intensité est encore multipliée par
        :meth:`_traffic_multiplier` (réponse endogène au prix et à la rareté).
        """
        horizon = self.initial_time
        eag = self._eagerness
        n = len(self.buyers)

        tt = np.arange(horizon, dtype=float)
        base_launch = np.exp(-tt / max(1e-6, VISIT_LAUNCH_TAU * horizon))
        base_event = np.exp(-(horizon - 1.0 - tt) / max(1e-6, VISIT_EVENT_TAU * horizon))
        base_launch /= base_launch.max()
        base_event /= base_event.max()
        self._base_launch = base_launch
        self._base_event = base_event

        is_late = self._is_late          # cohorte « dernière minute » (cf. _build_population)
        self._w_launch = 1.0 + VISIT_EAGER_LAUNCH_GAIN * eag
        self._w_event = (1.0 - eag) + VISIT_LATE_EVENT_GAIN * is_late
        self._w_floor = float(VISIT_FLOOR)

        # Normalisation : sum_t λ_i(t) == max_visits_i  (avant multiplicateur).
        self._visit_norm = (
            self._w_launch * base_launch.sum()
            + self._w_event * base_event.sum()
            + self._w_floor * horizon
        )
        self._active = np.ones(n, dtype=bool)

        # Baseline agrégée (multiplicateur == 1), pour le diagnostic / notebooks.
        k1 = float(np.sum(self._max_visits * self._w_launch / self._visit_norm))
        k2 = float(np.sum(self._max_visits * self._w_event / self._visit_norm))
        k3 = float(np.sum(self._max_visits * self._w_floor / self._visit_norm))
        self.expected_visits_by_step = base_launch * k1 + base_event * k2 + k3
        self.visits_this_step: int = 0

    # ------------------------------------------------------------------ #
    # Attentes du public : courbe de prix, croyance de sold-out           #
    # ------------------------------------------------------------------ #

    def expected_price(self, t: int | None = None) -> float:
        """Prix que le public s'attend à voir au pas ``t`` (défaut : pas courant).

        Ancre (fonction de la popularité) qui monte linéairement sur la fenêtre de
        vente : ``ancre · (1 + PRICE_ESCALATION_EXPECTED · t/horizon)``. Tarifer le
        long de cette courbe est « neutre » pour le trafic et pour l'attente.
        """
        step = self._step_index if t is None else t
        return self._price_anchor * (1.0 + PRICE_ESCALATION_EXPECTED * step / self.initial_time)

    def _traffic_multiplier(self, price: float) -> float:
        """Facteur endogène appliqué à l'intensité de visite du pas courant.

        > 1 quand le prix est sous la courbe de prix attendu (afflux de chasseurs
        de bonnes affaires) ou quand le stock fond vite près de la date (ruée) ;
        < 1 quand le prix dépasse la courbe attendue.
        """
        price_term = (self.expected_price() / max(price, 1e-6)) ** TRAFFIC_PRICE_ELASTICITY
        panic = max(0.0, self.sales_pace_ratio(window=5) - 1.0) * (1.0 - self.inventory_ratio)
        scarcity_term = 1.0 + TRAFFIC_SCARCITY_GAIN * panic
        return float(np.clip(price_term * scarcity_term, TRAFFIC_MULT_MIN, TRAFFIC_MULT_MAX))

    # ------------------------------------------------------------------ #
    # Boucle de simulation                                               #
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        if self.current_price is None:
            raise RuntimeError(
                "TicketMarketModel.current_price doit être défini avant step()."
            )

        price = float(self.current_price)
        self.tickets_sold_this_step = 0
        self.visits_this_step = 0
        if self._step_index < self.initial_time:
            self.urgency = 1.0 - self.time_remaining / self.initial_time
            t = self._step_index
            mult = self._traffic_multiplier(price)
            inst = (
                self._base_launch[t] * self._w_launch
                + self._base_event[t] * self._w_event
                + self._w_floor
            )
            lam = self._max_visits * inst / self._visit_norm * mult
            drawn = (self.rng.random(len(self.buyers)) < np.minimum(lam, 1.0)) & self._active
            buyers = self.buyers
            for idx in np.nonzero(drawn)[0]:
                buyers[idx].visit()
                self._active[idx] = buyers[idx].is_active
            self.visits_this_step = int(drawn.sum())

        self.sales_history.append(self.tickets_sold_this_step)
        self._step_index += 1
        self.time_remaining = max(0, self.initial_time - self._step_index)
        self.datacollector.collect(self)

    def attempt_purchase(self) -> bool:
        """Tentative de réservation d'un billet ; ``False`` si la salle est pleine."""
        if self.tickets_remaining > 0:
            self.tickets_remaining -= 1
            self.tickets_sold_this_step += 1
            self.tickets_sold_total += 1
            return True
        return False

    # ------------------------------------------------------------------ #
    # Observations                                                        #
    # ------------------------------------------------------------------ #

    @property
    def time_ratio(self) -> float:
        return self.time_remaining / self.initial_time

    @property
    def inventory_ratio(self) -> float:
        return self.tickets_remaining / self.initial_tickets

    @property
    def reference_price(self) -> float:
        """Alias de :meth:`expected_price` (prix attendu par le public au pas courant)."""
        return self.expected_price()

    @property
    def fill_rate(self) -> float:
        return self.tickets_sold_total / self.initial_tickets

    @property
    def done(self) -> bool:
        return self.tickets_remaining <= 0 or self.time_remaining <= 0

    def get_recent_sales(self, window: int = 1) -> int:
        if not self.sales_history:
            return 0
        return int(sum(self.sales_history[-window:]))

    def sales_pace_ratio(self, window: int = 5) -> float:
        """Rythme de vente récent rapporté au rythme « idéal » (linéaire).

        > 1 : on vend plus vite que le rythme nécessaire pour écouler le stock ;
        < 1 : on est en retard.
        """
        target_per_step = self.initial_tickets / self.initial_time
        if target_per_step <= 0:
            return 0.0
        window = min(window, len(self.sales_history)) or 1
        recent = self.get_recent_sales(window) / window
        return recent / target_per_step

    # -- rétrocompatibilité ------------------------------------------- #

    def get_time_ratio(self) -> float:
        return self.time_ratio

    def get_inventory_ratio(self) -> float:
        return self.inventory_ratio

    def get_artist_popularity(self) -> float:
        return self.popularity_index
