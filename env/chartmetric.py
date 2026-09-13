"""Client ChartMetric « cache-first ».

Règle : on ne contacte l'API **que si la donnée n'est pas déjà dans un CSV local**.
Tout ce qui est récupéré est aplati et écrit sous ``data/raw/chartmetric/`` :

    artists_index.csv          requête -> cm_id (+ nom résolu, spotify_id)   [résout /search]
    artist_metadata.csv        /artist/:id                (riche : ~130 colonnes)
    artist_stat.csv            /artist/:id/stat/:source   (long : source, field, timestamp, value)
    artist_geo.csv             /artist/:id/where-people-listen   (villes + pays, dernier point)
    artist_audience_age.csv    /artist/:id/:platform-audience-stats -> audience_genders_per_age
    artist_audience_gender.csv  idem -> audience_genders
    artist_audience_country.csv idem -> top_countries

La **clé de cache** est l'artiste (``cm_id``) : s'il figure déjà dans le CSV d'un
endpoint, l'API n'est pas rappelée pour cet endpoint. ``refresh=True`` force.

Auth : ``CHARTMETRIC_REFRESH_TOKEN`` — variable d'environnement, sinon fichier
``.env`` à la racine du projet. Le jeton d'accès (~1 h) est rafraîchi en mémoire
uniquement — aucun secret n'est écrit sur le disque.

Note : l'API artiste de ChartMetric **n'expose pas la liste des concerts / dates de
tournée** (tous les endpoints ``/artist/:id/concerts|tour|events`` renvoient 404).
Seul le lien Bandsintown et une série de followers sont disponibles via
``stat/bandsintown``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Emplacements & constantes                                                    #
# --------------------------------------------------------------------------- #

_REPO = Path(__file__).resolve().parents[1]
RAW_DIR = _REPO / "data" / "raw" / "chartmetric"
UPSERT_DIR = _REPO / "data" / "upsert"
BASE_URL = "https://api.chartmetric.com/api"

#: sources valides pour /artist/:id/stat/:source (d'après le message d'erreur de l'API)
STAT_SOURCES = ("spotify", "deezer", "instagram", "tiktok", "soundcloud", "youtube_artist", "bandsintown")
#: plateformes qui exposent la démographie d'audience (Spotify ne l'expose pas)
AUDIENCE_PLATFORMS = ("instagram", "tiktok", "youtube")
#: tranches d'âge cibles (alignées sur demographics.csv / le simulateur)
AGE_TARGET_BRACKETS = ("13-17", "18-24", "25-34", "35-44", "45_plus")


# --------------------------------------------------------------------------- #
# Token                                                                        #
# --------------------------------------------------------------------------- #


_ENV_LOADED = False


def _load_project_env() -> None:
    """Charge le ``.env`` de la racine du projet dans ``os.environ`` (une fois)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = _REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _read_refresh_token() -> str:
    _load_project_env()
    token = os.getenv("CHARTMETRIC_REFRESH_TOKEN")
    if not token:
        raise RuntimeError(
            "CHARTMETRIC_REFRESH_TOKEN introuvable — définir la variable d'environnement "
            "ou l'ajouter au fichier .env à la racine du projet (cf. .env.example)."
        )
    return token


# --------------------------------------------------------------------------- #
# Client HTTP                                                                  #
# --------------------------------------------------------------------------- #


class ChartmetricError(RuntimeError):
    pass


class _Unavailable(ChartmetricError):
    """Endpoint / paramètre indisponible pour cet artiste (400/404/422)."""


class ChartmetricClient:
    def __init__(
        self,
        refresh_token: str | None = None,
        *,
        min_interval: float = 0.6,
        timeout: float = 45.0,
    ) -> None:
        self._refresh_token = refresh_token or _read_refresh_token()
        self._min_interval = float(min_interval)
        self._timeout = float(timeout)
        self._token: str | None = None
        self._token_expiry = 0.0
        self._last_call = 0.0

    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        resp = requests.post(
            f"{BASE_URL}/token", json={"refreshtoken": self._refresh_token}, timeout=self._timeout
        )
        if not resp.ok:
            raise ChartmetricError(f"Authentification échouée ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        self._token = data["token"]
        self._token_expiry = time.time() + float(data.get("expires_in", 3600))
        return self._token

    def _throttle(self) -> None:
        delta = time.time() - self._last_call
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        self._last_call = time.time()

    def get(self, path: str, params: dict | None = None, *, _retry_auth: bool = True) -> Any:
        self._throttle()
        url = f"{BASE_URL}/{path.lstrip('/')}"
        for attempt in range(5):
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self._access_token()}"},
                params=params,
                timeout=self._timeout,
            )
            if resp.status_code == 429:
                time.sleep(min(float(resp.headers.get("Retry-After", 2 ** attempt)), 30.0))
                continue
            if resp.status_code == 401 and _retry_auth:
                self._token, _retry_auth = None, False
                continue
            if resp.status_code in (400, 404, 422):
                raise _Unavailable(f"{resp.status_code} {path}")
            if not resp.ok:
                raise ChartmetricError(f"{resp.status_code} sur {path}: {resp.text[:200]}")
            payload = resp.json()
            return payload.get("obj", payload)
        raise ChartmetricError(f"Trop de 429 sur {path}")

    def try_get(self, path: str, params: dict | None = None) -> Any | None:
        try:
            return self.get(path, params)
        except _Unavailable:
            return None


# --------------------------------------------------------------------------- #
# Cache CSV (une table par endpoint, clé = cm_id)                              #
# --------------------------------------------------------------------------- #


class CsvStore:
    def __init__(self, directory: str | Path = RAW_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.csv"

    def load(self, name: str) -> pd.DataFrame:
        p = self.path(name)
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def has(self, name: str, cm_id: int | str) -> bool:
        df = self.load(name)
        return not df.empty and "cm_id" in df.columns and int(cm_id) in set(pd.to_numeric(df["cm_id"], errors="coerce").dropna().astype(int))

    def replace_artist(self, name: str, cm_id: int | str, rows: pd.DataFrame) -> None:
        if rows is None or rows.empty:
            return
        rows = rows.copy()
        rows.insert(0, "cm_id", int(cm_id))
        rows["fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        old = self.load(name)
        if not old.empty and "cm_id" in old.columns:
            old = old[pd.to_numeric(old["cm_id"], errors="coerce") != int(cm_id)]
        pd.concat([old, rows], ignore_index=True).to_csv(self.path(name), index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# Aplatisseurs (formes réelles observées sur l'API)                            #
# --------------------------------------------------------------------------- #


def _flatten_stat(obj: Any, source: str) -> pd.DataFrame:
    rows: list[dict] = []
    if isinstance(obj, dict):
        for field, series in obj.items():
            if isinstance(series, list):
                for pt in series:
                    if isinstance(pt, dict):
                        rows.append(
                            {
                                "source": source,
                                "field": field,
                                "timestamp": pt.get("timestp") or pt.get("timestamp") or pt.get("date"),
                                "value": pt.get("value"),
                            }
                        )
    return pd.DataFrame(rows)


def _flatten_geo(obj: Any) -> pd.DataFrame:
    """``where-people-listen`` -> dernier point par ville et par pays."""
    rows: list[dict] = []
    _singular = {"cities": "city", "countries": "country"}
    if isinstance(obj, dict):
        for region_type, mapping in obj.items():          # "cities" / "countries"
            if not isinstance(mapping, dict):
                continue
            for name, series in mapping.items():
                if isinstance(series, list) and series:
                    last = series[-1]
                    rows.append(
                        {
                            "region_type": _singular.get(region_type, region_type),
                            "name": name,
                            "code2": last.get("code2"),
                            "listeners": last.get("listeners"),
                            "rank": last.get("artist_city_rank") or last.get("rank"),
                            "timestamp": last.get("timestp"),
                        }
                    )
    return pd.DataFrame(rows)


def _flatten_audience_age(obj: Any, platform: str) -> pd.DataFrame:
    block = obj.get("audience_genders_per_age") if isinstance(obj, dict) else None
    rows: list[dict] = []
    if isinstance(block, list):
        for item in block:
            m = float(item.get("male", 0) or 0)
            f = float(item.get("female", 0) or 0)
            rows.append(
                {"platform": platform, "age_code": item.get("code"), "male_pct": m, "female_pct": f, "total_pct": m + f}
            )
    return pd.DataFrame(rows)


def _flatten_audience_gender(obj: Any, platform: str) -> pd.DataFrame:
    block = obj.get("audience_genders") if isinstance(obj, dict) else None
    rows: list[dict] = []
    if isinstance(block, list):
        for item in block:
            rows.append({"platform": platform, "gender": item.get("code"), "weight_pct": float(item.get("weight", 0) or 0)})
    return pd.DataFrame(rows)


def _flatten_audience_country(obj: Any, platform: str) -> pd.DataFrame:
    block = obj.get("top_countries") if isinstance(obj, dict) else None
    rows: list[dict] = []
    if isinstance(block, list):
        for item in block:
            rows.append(
                {
                    "platform": platform,
                    "country": item.get("name"),
                    "code": item.get("code"),
                    "percent": float(item.get("percent", 0) or 0),
                    "followers": item.get("followers") or item.get("subscribers"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Résolution d'artiste                                                         #
# --------------------------------------------------------------------------- #


def resolve_artist(client: ChartmetricClient, name: str, store: CsvStore, *, refresh: bool = False) -> dict | None:
    idx = store.load("artists_index")
    if not refresh and not idx.empty and "query" in idx.columns:
        hit = idx.loc[idx["query"].astype(str).str.casefold() == name.casefold()]
        if not hit.empty:
            r = hit.iloc[0]
            return {"cm_id": int(r["cm_id"]), "name": r["name"]}

    res = client.get("search", params={"q": name, "type": "artists", "limit": 1})
    artists = (res or {}).get("artists") or []
    if not artists:
        print(f"  [!] '{name}' introuvable sur ChartMetric")
        return None
    a = artists[0]
    sp = a.get("spotify_artist_ids") or a.get("spotify_artist_id")
    entry = {
        "query": name,
        "cm_id": int(a["id"]),
        "name": a.get("name", name),
        "spotify_id": sp[0] if isinstance(sp, list) and sp else sp,
    }
    if not idx.empty and "query" in idx.columns:
        idx = idx[idx["query"].astype(str).str.casefold() != name.casefold()]
    pd.concat([idx, pd.DataFrame([entry])], ignore_index=True).to_csv(
        store.path("artists_index"), index=False, encoding="utf-8-sig"
    )
    return {"cm_id": entry["cm_id"], "name": entry["name"]}


# --------------------------------------------------------------------------- #
# Récupération complète d'un artiste                                           #
# --------------------------------------------------------------------------- #


def fetch_artist(
    client: ChartmetricClient, name: str, *, store: CsvStore | None = None, refresh: bool = False
) -> int | None:
    store = store or CsvStore()
    resolved = resolve_artist(client, name, store, refresh=refresh)
    if resolved is None:
        return None
    cm_id = resolved["cm_id"]
    print(f"- {resolved['name']} (cm_id={cm_id})")

    if refresh or not store.has("artist_metadata", cm_id):
        meta = client.get(f"artist/{cm_id}")
        flat = pd.json_normalize(meta, sep="_") if isinstance(meta, dict) else pd.DataFrame()
        for col in flat.columns:
            if flat[col].apply(lambda x: isinstance(x, (list, dict))).any():
                flat[col] = flat[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
                )
        store.replace_artist("artist_metadata", cm_id, flat)
        print("    métadonnées ✓")

    if refresh or not store.has("artist_stat", cm_id):
        frames = [
            _flatten_stat(d, src)
            for src in STAT_SOURCES
            if (d := client.try_get(f"artist/{cm_id}/stat/{src}")) is not None
        ]
        frames = [f for f in frames if not f.empty]
        if frames:
            store.replace_artist("artist_stat", cm_id, pd.concat(frames, ignore_index=True))
            print("    stats séries temporelles ✓")

    if refresh or not store.has("artist_geo", cm_id):
        geo = _flatten_geo(client.try_get(f"artist/{cm_id}/where-people-listen"))
        if not geo.empty:
            store.replace_artist("artist_geo", cm_id, geo)
            print("    géographie d'écoute ✓")

    if refresh or not store.has("artist_audience_age", cm_id):
        age_f, gen_f, cty_f = [], [], []
        for platform in AUDIENCE_PLATFORMS:
            d = client.try_get(f"artist/{cm_id}/{platform}-audience-stats")
            if d:
                age_f.append(_flatten_audience_age(d, platform))
                gen_f.append(_flatten_audience_gender(d, platform))
                cty_f.append(_flatten_audience_country(d, platform))
        age = pd.concat([f for f in age_f if not f.empty], ignore_index=True) if any(not f.empty for f in age_f) else pd.DataFrame()
        if not age.empty:
            store.replace_artist("artist_audience_age", cm_id, age)
            store.replace_artist("artist_audience_gender", cm_id, pd.concat([f for f in gen_f if not f.empty], ignore_index=True))
            store.replace_artist("artist_audience_country", cm_id, pd.concat([f for f in cty_f if not f.empty], ignore_index=True))
            print("    démographie d'audience (âge / genre / pays) ✓")

    return cm_id


def fetch_many(names: Iterable[str], *, refresh: bool = False, min_interval: float = 0.6) -> list[int]:
    client = ChartmetricClient(min_interval=min_interval)
    store = CsvStore()
    ids: list[int] = []
    for name in names:
        try:
            cid = fetch_artist(client, name, store=store, refresh=refresh)
            if cid is not None:
                ids.append(cid)
        except ChartmetricError as exc:
            print(f"  [ERREUR] {name}: {exc}")
    return ids


# --------------------------------------------------------------------------- #
# Construction des CSV « curés » consommés par le simulateur                    #
# --------------------------------------------------------------------------- #

_GENRE_AGE_FALLBACK: dict[str, tuple[float, ...]] = {
    "metal": (0.05, 0.20, 0.40, 0.25, 0.10),
    "rock": (0.05, 0.20, 0.40, 0.25, 0.10),
    "pop": (0.25, 0.45, 0.20, 0.08, 0.02),
    "rap": (0.25, 0.45, 0.20, 0.08, 0.02),
    "indie": (0.10, 0.35, 0.35, 0.15, 0.05),
    "jazz": (0.10, 0.30, 0.30, 0.20, 0.10),
    "classical": (0.10, 0.30, 0.30, 0.20, 0.10),
}
_GENRE_AGE_DEFAULT = (0.10, 0.30, 0.30, 0.20, 0.10)

#: rapproche un genre ChartMetric (souvent très fin) d'un genre de genres.csv
_GENRE_ALIASES: dict[str, str] = {
    "soundtrack": "Classical", "us soundtrack": "Classical", "score": "Classical",
    "orchestral": "Classical", "film music": "Classical", "modern classical": "Classical",
    "neoclassical": "Classical", "contemporary classical": "Classical",
    "heavy metal": "Metal", "power metal": "Metal", "melodic metal": "Metal",
    "symphonic metal": "Metal", "medieval metal": "Metal", "metalcore": "Metal",
    "hard rock": "Metal", "nu metal": "Metal",
    "contemporary jazz": "Jazz", "modern jazz": "Jazz", "jazz fusion": "Jazz", "bebop": "Jazz",
    "indie rock": "Indie", "indie pop": "Indie", "bedroom pop": "Indie",
    "dance pop": "Pop", "electropop": "Pop", "art pop": "Pop",
}


def _map_age_bracket(raw: str) -> str | None:
    digits = [int(x) for x in "".join(c if c.isdigit() else " " for c in str(raw)).split()]
    if not digits:
        return None
    lo = digits[0]
    hi = digits[1] if len(digits) > 1 else lo + 5
    mid = (lo + hi) / 2
    if mid < 18:
        return "13-17"
    if mid < 25:
        return "18-24"
    if mid < 35:
        return "25-34"
    if mid < 45:
        return "35-44"
    return "45_plus"


def _age_distribution(cm_id: int, store: CsvStore, genre: str) -> tuple[float, ...]:
    df = store.load("artist_audience_age")
    if not df.empty and "cm_id" in df.columns:
        sub = df[pd.to_numeric(df["cm_id"], errors="coerce") == int(cm_id)]
        if not sub.empty:
            sub = sub.assign(bracket=sub["age_code"].map(_map_age_bracket)).dropna(subset=["bracket"])
            agg = sub.groupby("bracket")["total_pct"].mean()  # moyenne des plateformes
            vec = [float(agg.get(b, 0.0)) for b in AGE_TARGET_BRACKETS]
            if sum(vec) > 0:
                return tuple(v / sum(vec) for v in vec)
    return _GENRE_AGE_FALLBACK.get(str(genre).casefold(), _GENRE_AGE_DEFAULT)


def _pick_genre(meta_row: pd.Series, known_genres: set[str]) -> str:
    """Rapproche les genres ChartMetric d'un genre présent dans genres.csv."""
    candidates: list[str] = []
    for col in ("genres_primary_name", "genres_secondary", "genres_sub"):
        raw = meta_row.get(col)
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                candidates += [g["name"] for g in json.loads(raw) if "name" in g]
            except json.JSONDecodeError:
                pass
        elif isinstance(raw, str) and raw:
            candidates.append(raw)

    for c in candidates:                       # 1. correspondance exacte
        if c and c.casefold() in known_genres:
            return c.title()
    for c in candidates:                       # 2. alias connu
        if c and c.casefold() in _GENRE_ALIASES:
            return _GENRE_ALIASES[c.casefold()]
    for c in candidates:                       # 3. mot-clé contenu
        for kw, target in _GENRE_ALIASES.items():
            if kw in c.casefold():
                return target
    return candidates[0].title() if candidates and candidates[0] else "Unknown"


def rebuild_artists_csv(
    store: CsvStore | None = None,
    out: str | Path | None = None,
    *,
    merge: bool = True,
) -> pd.DataFrame:
    """Régénère ``data/upsert/artists.csv`` à partir du cache ChartMetric.

    - audience / followers / popularité : colonnes ``cm_statistics_*`` des métadonnées
    - distribution d'âge : démographie d'audience réelle, sinon estimation par genre
    - ``cm_artist_score`` : score artiste ChartMetric (indice de popularité alternatif)

    ``merge=True`` (défaut) conserve les artistes déjà présents dans le CSV mais
    absents du cache, au lieu de les écraser.
    """
    store = store or CsvStore()
    meta = store.load("artist_metadata")
    if meta.empty:
        raise RuntimeError("artist_metadata.csv vide — lancer scripts/fetch_chartmetric.py d'abord.")

    genres_csv = UPSERT_DIR / "genres.csv"
    known = set(pd.read_csv(genres_csv)["genre_name"].str.casefold()) if genres_csv.exists() else set()

    def val(row: pd.Series, *names: str, default: Any = 0) -> Any:
        for n in names:
            if n in row and pd.notna(row[n]):
                return row[n]
        return default

    rows = []
    for _, m in meta.iterrows():
        cm_id = int(m["cm_id"])
        name = val(m, "name", default=f"cm_{cm_id}")
        genre = _pick_genre(m, known)
        ages = _age_distribution(cm_id, store, genre)
        rows.append(
            {
                "artist_name": name,
                "primary_genre": genre,
                "global_monthly_listeners": int(float(val(m, "cm_statistics_sp_monthly_listeners"))),
                "total_followers": int(float(val(m, "cm_statistics_sp_followers"))),
                "spotify_popularity": float(val(m, "cm_statistics_sp_popularity")),
                "cm_artist_score": round(float(val(m, "cm_artist_score", "cm_statistics_cm_artist_score")), 2),
                "age_13_17_pct": round(ages[0], 4),
                "age_18_24_pct": round(ages[1], 4),
                "age_25_34_pct": round(ages[2], 4),
                "age_35_44_pct": round(ages[3], 4),
                "age_45_plus_pct": round(ages[4], 4),
            }
        )

    df = pd.DataFrame(rows)
    out = Path(out) if out else UPSERT_DIR / "artists.csv"

    if merge and out.exists():
        existing = pd.read_csv(out)
        kept = existing[~existing["artist_name"].str.casefold().isin(df["artist_name"].str.casefold())]
        df = pd.concat([kept, df], ignore_index=True)

    df = df.sort_values("artist_name").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{out} régénéré ({len(df)} artistes ; {len(rows)} depuis le cache ChartMetric)")
    return df


def _engagement_to_genre_params(ratio: float) -> tuple[float, float]:
    """Ratio abonnés/auditeurs -> (taux de conversion de base, facteur de mobilité)."""
    if ratio >= 0.5:
        return 0.15, 1.5
    if ratio >= 0.3:
        return 0.10, 1.2
    return 0.03, 1.0


def rebuild_genres_csv(
    store: CsvStore | None = None, out: str | Path | None = None, *, merge: bool = True
) -> pd.DataFrame:
    """Régénère ``data/upsert/genres.csv`` : conversion / mobilité par genre,
    à partir de l'engagement moyen (abonnés / auditeurs) des artistes en cache."""
    store = store or CsvStore()
    meta = store.load("artist_metadata")
    if meta.empty:
        raise RuntimeError("artist_metadata.csv vide.")

    out = Path(out) if out else UPSERT_DIR / "genres.csv"
    known = set(pd.read_csv(out)["genre_name"].str.casefold()) if out.exists() else set()

    genre_col = meta.apply(lambda m: _pick_genre(m, known), axis=1)
    eng_col = pd.to_numeric(meta.get("cm_statistics_sp_followers"), errors="coerce") / pd.to_numeric(
        meta.get("cm_statistics_sp_monthly_listeners"), errors="coerce"
    )
    agg = eng_col.groupby(genre_col).mean().clip(0, 1)

    rows = []
    for genre, ratio in agg.items():
        conv, mob = _engagement_to_genre_params(float(ratio) if pd.notna(ratio) else 0.0)
        rows.append({"genre_name": genre, "base_conversion_rate_pct": conv, "mobility_factor": mob})
    df = pd.DataFrame(rows)

    if merge and out.exists():
        existing = pd.read_csv(out)
        kept = existing[~existing["genre_name"].str.casefold().isin(df["genre_name"].str.casefold())]
        df = pd.concat([kept, df], ignore_index=True)
    if "unknown" not in df["genre_name"].str.casefold().values:
        df = pd.concat([df, pd.DataFrame([{"genre_name": "Unknown", "base_conversion_rate_pct": 0.05, "mobility_factor": 1.0}])], ignore_index=True)

    df = df.sort_values("genre_name").reset_index(drop=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{out} régénéré ({len(df)} genres)")
    return df
