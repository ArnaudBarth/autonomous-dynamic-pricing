"""Chargement des données de référence qui pilotent le simulateur de marché.

Deux sources alimentent l'environnement :

* **ChartMetric** (``data/upsert/artists.csv``) — pour chaque artiste : audience,
  nombre d'abonnés, score de popularité Spotify et distribution d'âge du public.
  Ces chiffres déterminent *combien* d'acheteurs virtuels créer et *de quel âge*.
* **OFS** (``data/upsert/demographics.csv``) — par tranche d'âge : salaire mensuel
  brut médian, taux de participation aux concerts et volatilité budgétaire.
  Ces chiffres déterminent la *plage de budget* de chaque acheteur.

``genres.csv`` (fidélité / conversion par style) et ``venues.csv`` (capacité de
salle, Fan Cost Index) complètent le tableau.

Toute la logique de calibration (revenu -> budget, genre + fidélité -> intérêt,
popularité + style -> sensibilité au prix) est regroupée ici afin que
``market.py`` et ``agents.py`` restent lisibles.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Emplacement des fichiers                                                     #
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "upsert"

AGE_BRACKETS: tuple[str, ...] = ("13-17", "18-24", "25-34", "35-44", "45_plus")

_ARTIST_AGE_COLUMNS = {
    "13-17": "age_13_17_pct",
    "18-24": "age_18_24_pct",
    "25-34": "age_25_34_pct",
    "35-44": "age_35_44_pct",
    "45_plus": "age_45_plus_pct",
}

# --------------------------------------------------------------------------- #
# Constantes de calibration (documentées pour le rapport)                      #
# --------------------------------------------------------------------------- #

#: Budget = *plafond de disposition à payer* pour un concert désiré — PAS la
#: dépense culturelle moyenne de l'OFS (~8-10 CHF, dominée par les non-pratiquants).
#: On le dérive du **revenu discrétionnaire** (revenu brut moins un forfait de
#: besoins essentiels), modulé par l'**intensité de pratique** de la tranche d'âge
#: (participation × nombre de concerts/an, données OFS). Le facteur de part est une
#: hypothèse de modélisation, à discuter dans le rapport (analyse de sensibilité).
SUBSISTENCE_INCOME_RATE = 0.55          #: part du revenu couvrant les besoins essentiels
DISCRETIONARY_TICKET_SHARE = 0.045      #: part du revenu discrétionnaire pour UN billet voulu

#: Pression budgétaire (bêta_b) : pénalité d'utilité **lisse** appliquée quand le
#: prix dépasse ``BUDGET_COMFORT`` × budget (remplace l'ancien plafond dur, qui
#: faisait double emploi avec la sensibilité au prix bêta_p). Un agent très motivé
#: peut encore acheter légèrement au-dessus de son budget.
BUDGET_COMFORT = 0.70
BUDGET_PRESSURE = 6.0
BUDGET_PRESSURE_NOISE_SD = 1.5

#: Intérêt brut (bêta_0) = INTEREST_BASE
#:                        + INTEREST_GENRE_GAIN  * taux_conversion_genre
#:                        + INTEREST_LOYALTY_GAIN * ratio_d_engagement
INTEREST_BASE = -0.55
INTEREST_GENRE_GAIN = 6.0
INTEREST_LOYALTY_GAIN = 1.5
INTEREST_NOISE_SD = 0.40

#: Sensibilité au prix (bêta_p, CHF⁻¹). Part d'une valeur de base puis diminue
#: avec la popularité de l'artiste et la "mobilité" du style : les fans d'une
#: superstar métal sont peu regardants, le grand public d'un artiste pop l'est.
PRICE_SENS_BASE = 0.055
PRICE_SENS_POP_DROP = 0.015         # par point d'indice de popularité au-dessus de 1
PRICE_SENS_STYLE_DROP = 0.010       # par point de mobility_factor au-dessus de 1
PRICE_SENS_NOISE_SD = 0.006
PRICE_SENS_MIN = 0.015
PRICE_SENS_MAX = 0.090

#: Prime d'urgence (bêta_u) : gain d'utilité quand on passe du lancement de la
#: billetterie (urgence 0) à la veille de l'évènement (urgence 1). Les acheteurs
#: patients (peu "empressés") révèlent une disposition à payer plus forte tard.
URGENCY_BASE = 0.55
URGENCY_PATIENCE_GAIN = 1.80
URGENCY_NOISE_SD = 0.20

#: Cohorte « dernière minute » (part = VISIT_LATE_FRACTION, tirée chez les peu
#: empressés) : ces acheteurs se décident tard (agenda incertain) mais leur
#: disposition à payer est alors forte — segment qui justifie de ne pas brader tôt.
LATE_INTEREST_BONUS = 0.8          #: + sur bêta_0
LATE_PRICE_SENS_FACTOR = 0.65      #: × sur bêta_p (moins regardants une fois décidés)
LATE_URGENCY_BONUS = 0.6           #: + sur bêta_u

#: Nombre moyen de visites de la plateforme par acheteur potentiel avant l'achat
#: ou l'abandon. Modulé par la mobilité du style musical.
VISITS_BASE = 1.4
VISITS_STYLE_GAIN = 0.8

#: Demande potentielle = capacité * (DEMAND_RATIO_BASE
#:                                   + DEMAND_RATIO_POP_GAIN * indice_popularité)
#: bruitée multiplicativement (fort bruit : le prix optimal dépend beaucoup du
#: tirage de demande, que l'agent doit inférer du rythme de vente en cours de route).
DEMAND_RATIO_BASE = 0.58
DEMAND_RATIO_POP_GAIN = 0.24
DEMAND_NOISE_SD = 0.32
DEMAND_MIN_BUYERS = 25
DEMAND_MAX_RATIO = 3.2             #: plafond de la demande (× capacité) — perf + réalisme

# --------------------------------------------------------------------------- #
# Processus d'arrivée des visites (Poisson non homogène, intensité endogène)   #
# --------------------------------------------------------------------------- #
#: Forme temporelle de l'intensité de visite : somme de trois profils normalisés
#: — pic à l'ouverture de la billetterie, plancher de trafic constant, rappel
#: accéléré avant la date. Constantes de temps en fraction de l'horizon.
VISIT_LAUNCH_TAU = 0.22
VISIT_EVENT_TAU = 0.11
VISIT_FLOOR = 0.20                 #: plancher de trafic (poids relatif au pic, dont le max = 1)
VISIT_EAGER_LAUNCH_GAIN = 1.5      #: poids du pic d'ouverture ≈ 1 + gain · empressement
VISIT_LATE_FRACTION = 0.28         #: part d'acheteurs « dernière minute »
VISIT_LATE_EVENT_GAIN = 7.0       #: amplification du rappel pré-évènement pour cette cohorte

# --------------------------------------------------------------------------- #
# Courbe de prix attendu par le public (référence du multiplicateur de trafic) #
# --------------------------------------------------------------------------- #
#: prix_attendu(t) = ancre · (1 + PRICE_ESCALATION_EXPECTED · t/horizon)
#: L'ancre dépend de la popularité (une superstar « vaut » plus cher d'emblée) et
#: monte doucement sur la fenêtre. Tarifer le long de cette courbe est « neutre »
#: pour le trafic ; en dessous attire les chasseurs de deal, au-dessus le raréfie.
PRICE_ANCHOR_BASE = 42.0
PRICE_ANCHOR_POP_SLOPE = 5.5       #: CHF par point de popularité au-dessus de 1
PRICE_ANCHOR_NOISE_SD = 0.12       #: bruit log-normal (randomisation de domaine)
PRICE_ESCALATION_EXPECTED = 0.30   #: hausse anticipée sur toute la fenêtre de vente

# --------------------------------------------------------------------------- #
# Endogénéité du trafic (Poisson non homogène)                                 #
# --------------------------------------------------------------------------- #
#:   M(t) = clip( (prix_attendu(t) / prix) ** TRAFFIC_PRICE_ELASTICITY
#:                · (1 + TRAFFIC_SCARCITY_GAIN · panique), MULT_MIN, MULT_MAX )
#: panique = max(0, rythme_de_vente − 1) · (1 − inventaire_restant).
TRAFFIC_PRICE_ELASTICITY = 0.8
TRAFFIC_SCARCITY_GAIN = 2.5
TRAFFIC_MULT_MIN = 0.25
TRAFFIC_MULT_MAX = 4.0


# --------------------------------------------------------------------------- #
# Chargement des CSV (mis en cache)                                            #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=8)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_artists(data_dir: str | Path | None = None) -> pd.DataFrame:
    return _read_csv(str(Path(data_dir or DATA_DIR) / "artists.csv"))


def load_demographics(data_dir: str | Path | None = None) -> pd.DataFrame:
    return _read_csv(str(Path(data_dir or DATA_DIR) / "demographics.csv"))


def load_genres(data_dir: str | Path | None = None) -> pd.DataFrame:
    return _read_csv(str(Path(data_dir or DATA_DIR) / "genres.csv"))


def load_venues(data_dir: str | Path | None = None) -> pd.DataFrame:
    return _read_csv(str(Path(data_dir or DATA_DIR) / "venues.csv"))


# --------------------------------------------------------------------------- #
# Profils                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenreProfile:
    name: str
    base_conversion_rate: float
    mobility_factor: float

    @classmethod
    def lookup(cls, genre: str, data_dir: str | Path | None = None) -> "GenreProfile":
        genres = load_genres(data_dir)
        match = genres.loc[genres["genre_name"].str.casefold() == str(genre).casefold()]
        if match.empty:
            match = genres.loc[genres["genre_name"] == "Unknown"]
        row = match.iloc[0]
        return cls(
            name=str(row["genre_name"]),
            base_conversion_rate=float(row["base_conversion_rate_pct"]),
            mobility_factor=float(row["mobility_factor"]),
        )


@dataclass(frozen=True)
class DemographicProfile:
    """Budget-billet par tranche d'âge, dérivé du revenu et de la pratique OFS."""

    #: bracket -> plafond de disposition à payer central (CHF)
    central_budget: dict[str, float]
    #: bracket -> volatilité relative du budget (écart triangulaire)
    volatility: dict[str, float]
    #: bracket -> part de la tranche qui fréquente les concerts (0–1)
    participation: dict[str, float]
    #: bracket -> intensité de pratique (participation × concerts/an), normalisée à moyenne 1
    concert_engagement: dict[str, float]

    @classmethod
    def from_csv(cls, data_dir: str | Path | None = None) -> "DemographicProfile":
        demo = load_demographics(data_dir).set_index("age_bracket")
        part = {b: float(demo.loc[b, "concert_participation_ratio_pct"]) / 100.0 for b in AGE_BRACKETS}
        freq = {b: float(demo.loc[b, "annual_concert_quantity"]) for b in AGE_BRACKETS}
        raw_eng = {b: part[b] * freq[b] for b in AGE_BRACKETS}
        mean_eng = sum(raw_eng.values()) / len(raw_eng)
        engagement = {b: raw_eng[b] / mean_eng for b in AGE_BRACKETS}

        central, vol = {}, {}
        for b in AGE_BRACKETS:
            disposable = float(demo.loc[b, "median_monthly_gross_chf"]) * (1.0 - SUBSISTENCE_INCOME_RATE)
            central[b] = disposable * DISCRETIONARY_TICKET_SHARE * engagement[b]
            vol[b] = float(demo.loc[b, "budget_volatility_pct"])
        return cls(central, vol, part, engagement)

    def sample_budgets(self, brackets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Tire un budget (CHF) pour chaque acheteur selon sa tranche d'âge."""
        central = np.array([self.central_budget[b] for b in brackets])
        vol = np.array([self.volatility[b] for b in brackets])
        low = central * (1.0 - vol)
        high = central * (1.0 + vol)
        return rng.triangular(low, central, high)


@dataclass(frozen=True)
class ArtistProfile:
    name: str
    genre: str
    monthly_listeners: int
    followers: int
    spotify_popularity: float
    age_distribution: dict[str, float]

    # -- indicateurs dérivés ------------------------------------------------ #

    @property
    def engagement_ratio(self) -> float:
        """Abonnés / auditeurs mensuels : proxy de la fidélité du public."""
        if self.monthly_listeners <= 0:
            return 0.0
        return min(1.0, self.followers / self.monthly_listeners)

    @property
    def popularity_index(self) -> float:
        """Indice de popularité continu sur [1, 10].

        Combine l'audience (échelle log : 100 k -> 0, 100 M -> 1) et le score de
        popularité Spotify (0–100). Powerwolf ~= 5.2, Hans Zimmer ~= 7.6.
        """
        audience = np.log10(max(self.monthly_listeners, 1.0))
        audience_score = np.clip((audience - 5.0) / 3.0, 0.0, 1.0)
        spotify_score = np.clip(self.spotify_popularity / 100.0, 0.0, 1.0)
        combined = 0.6 * audience_score + 0.4 * spotify_score
        return float(1.0 + 9.0 * combined)

    # -- construction ----------------------------------------------------- #

    @classmethod
    def from_row(cls, row: pd.Series) -> "ArtistProfile":
        return cls(
            name=str(row["artist_name"]),
            genre=str(row["primary_genre"]),
            monthly_listeners=int(row["global_monthly_listeners"]),
            followers=int(row["total_followers"]),
            spotify_popularity=float(row["spotify_popularity"]),
            age_distribution={
                bracket: float(row[col]) for bracket, col in _ARTIST_AGE_COLUMNS.items()
            },
        )

    @classmethod
    def by_name(cls, name: str, data_dir: str | Path | None = None) -> "ArtistProfile":
        artists = load_artists(data_dir)
        match = artists.loc[artists["artist_name"].str.casefold() == str(name).casefold()]
        if match.empty:
            raise KeyError(f"Artiste introuvable dans artists.csv : {name!r}")
        return cls.from_row(match.iloc[0])

    @classmethod
    def sample(cls, rng: np.random.Generator, data_dir: str | Path | None = None) -> "ArtistProfile":
        artists = load_artists(data_dir)
        return cls.from_row(artists.iloc[int(rng.integers(len(artists)))])

    # -- distribution d'âge effective ----------------------------------- #

    def effective_age_weights(self, demo: DemographicProfile) -> dict[str, float]:
        """Distribution d'âge pondérée par la participation aux concerts."""
        weights = {
            b: self.age_distribution.get(b, 0.0) * demo.participation[b] for b in AGE_BRACKETS
        }
        total = sum(weights.values()) or 1.0
        return {b: w / total for b, w in weights.items()}


@dataclass(frozen=True)
class VenueProfile:
    name: str
    city: str
    capacity: int
    fan_cost_index: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "VenueProfile":
        return cls(
            name=str(row["venue_name"]),
            city=str(row["city"]),
            capacity=int(row["max_capacity"]),
            fan_cost_index=float(row["fan_cost_index_chf"]),
        )

    @classmethod
    def by_name(cls, name: str, data_dir: str | Path | None = None) -> "VenueProfile":
        venues = load_venues(data_dir)
        match = venues.loc[venues["venue_name"].str.casefold() == str(name).casefold()]
        if match.empty:
            raise KeyError(f"Salle introuvable dans venues.csv : {name!r}")
        return cls.from_row(match.iloc[0])

    @classmethod
    def sample(cls, rng: np.random.Generator, data_dir: str | Path | None = None) -> "VenueProfile":
        venues = load_venues(data_dir)
        return cls.from_row(venues.iloc[int(rng.integers(len(venues)))])

    @classmethod
    def sample_for_popularity(
        cls,
        rng: np.random.Generator,
        popularity_index: float,
        data_dir: str | Path | None = None,
        spread: float = 0.7,
    ) -> "VenueProfile":
        """Tire une salle dont la capacité est *cohérente* avec la popularité.

        Évite les scénarios irréalistes (artiste de niche dans un stade) qui
        fausseraient la comparaison des agents. Capacité cible ~= 400 places par
        point de popularité ; pondération log-normale autour de cette cible.
        """
        venues = load_venues(data_dir)
        caps = venues["max_capacity"].to_numpy(dtype=float)
        target = 400.0 * popularity_index
        weights = np.exp(-0.5 * ((np.log(caps) - np.log(target)) / spread) ** 2)
        weights /= weights.sum()
        return cls.from_row(venues.iloc[int(rng.choice(len(venues), p=weights))])
