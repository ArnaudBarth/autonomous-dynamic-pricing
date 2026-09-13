"""Versionnage des modèles entraînés (Q-table, DQN).

Chaque entraînement sauvegarde un fichier **horodaté** plus un sidecar
``.meta.json`` (date de création + hyperparamètres) :

    models/pricingAgents/q_table_20260828-181530.npy
    models/pricingAgents/q_table_20260828-181530.npy.meta.json
    models/pricingAgents/dqn_20260829-094512.keras
    models/pricingAgents/dqn_20260829-094512.keras.meta.json

Chargement :
    resolve_model("q_table")                    -> le plus récent
    resolve_model("q_table", "20260828-181530") -> une version précise (fragment du nom)
    resolve_model("q_table", "chemin/complet")   -> ce fichier
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "pricingAgents"

_EXT = {"q_table": ".npy", "dqn": ".keras"}
_TS_RE = re.compile(r"_(\d{8}-\d{6})(?=\.|$)")


def timestamp(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).strftime("%Y%m%d-%H%M%S")


def _dir(models_dir: str | Path | None) -> Path:
    return Path(models_dir) if models_dir else MODELS_DIR


def model_path(kind: str, dt: datetime | None = None, models_dir: str | Path | None = None) -> Path:
    """Chemin horodaté pour un nouveau modèle."""
    return _dir(models_dir) / f"{kind}_{timestamp(dt)}{_EXT[kind]}"


def checkpoint_path(kind: str, models_dir: str | Path | None = None) -> Path:
    """Emplacement (stable, ignoré par git) du checkpoint de reprise.

    Un seul entraînement en cours à la fois par type de modèle.
    """
    return _dir(models_dir) / f".checkpoint_{kind}{_EXT[kind]}"


def list_versions(kind: str, models_dir: str | Path | None = None) -> list[Path]:
    """Modèles horodatés de ce type, du plus ancien au plus récent."""
    d = _dir(models_dir)
    files = [p for p in d.glob(f"{kind}_*{_EXT[kind]}") if _TS_RE.search(p.stem + p.suffix)]
    return sorted(files, key=lambda p: _TS_RE.search(p.name).group(1))


def resolve_model(kind: str, version: str | None = None, models_dir: str | Path | None = None) -> Path:
    d = _dir(models_dir)
    if version:
        p = Path(version)
        if p.exists():
            return p
        hits = sorted(x for x in d.glob(f"{kind}_*{_EXT[kind]}") if version in x.name)
        if hits:
            return hits[-1]
        raise FileNotFoundError(f"Aucun modèle {kind} correspondant à {version!r} dans {d}")

    versions = list_versions(kind, d)
    if versions:
        return versions[-1]
    # repli : ancien nommage non horodaté (q_table_v1.npy, dqn_v1.keras)
    legacy = sorted(d.glob(f"{kind}_*{_EXT[kind]}"), key=lambda p: p.stat().st_mtime)
    if legacy:
        return legacy[-1]
    raise FileNotFoundError(f"Aucun modèle {kind} dans {d} — lancer l'entraînement d'abord.")


def write_meta(path: str | Path, **meta) -> None:
    meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    Path(f"{path}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def read_meta(path: str | Path) -> dict:
    mp = Path(f"{path}.meta.json")
    return json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
