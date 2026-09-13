"""Récupère (avec cache CSV) les données ChartMetric puis régénère les CSV du simulateur.

    python scripts/fetch_chartmetric.py                     # artistes de data/chartmetric_artists.txt
    python scripts/fetch_chartmetric.py --artists "Taylor Swift, Powerwolf"
    python scripts/fetch_chartmetric.py --refresh           # force le rappel de l'API
    python scripts/fetch_chartmetric.py --no-rebuild        # ne touche pas data/upsert/artists.csv

L'API ChartMetric n'est appelée que pour les artistes / endpoints absents du
cache (``data/raw/chartmetric/*.csv``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "env"))

from chartmetric import CsvStore, fetch_many, rebuild_artists_csv, rebuild_genres_csv  # noqa: E402

# NOTE : genres.csv (conversion / mobilité par genre) est **maintenu à la main** —
# le proxy « ratio abonnés/auditeurs » discrimine mal les genres quand les artistes
# vont de 5 k à 100 M d'auditeurs. `--rebuild-genres` force la régénération auto.

REPO = Path(__file__).resolve().parents[1]
SEED_FILE = REPO / "data" / "chartmetric_artists.txt"
ARTISTS_CSV = REPO / "data" / "upsert" / "artists.csv"


def default_artist_names() -> list[str]:
    if SEED_FILE.exists():
        return [
            line.strip()
            for line in SEED_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if ARTISTS_CSV.exists():
        import pandas as pd

        return pd.read_csv(ARTISTS_CSV)["artist_name"].tolist()
    raise SystemExit(
        f"Aucune source d'artistes : créez {SEED_FILE} (un nom par ligne) ou passez --artists."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artists", help="liste séparée par des virgules (sinon data/chartmetric_artists.txt)")
    p.add_argument("--refresh", action="store_true", help="rappelle l'API même si déjà en cache")
    p.add_argument("--rebuild", dest="rebuild", action="store_true", default=None,
                   help="régénère data/upsert/artists.csv depuis le cache")
    p.add_argument("--no-rebuild", dest="rebuild", action="store_false",
                   help="ne pas régénérer artists.csv")
    p.add_argument("--rebuild-genres", action="store_true",
                   help="régénère aussi genres.csv (déconseillé, cf note en tête de fichier)")
    p.add_argument("--interval", type=float, default=0.6, help="délai mini entre appels API (s)")
    args = p.parse_args()

    names = (
        [n.strip() for n in args.artists.split(",") if n.strip()]
        if args.artists
        else default_artist_names()
    )
    # Par défaut : rebuild uniquement quand on traite la liste complète (pas --artists),
    # pour éviter d'écraser artists.csv avec un sous-ensemble.
    do_rebuild = args.rebuild if args.rebuild is not None else (args.artists is None)

    print(f"{len(names)} artiste(s) : {', '.join(names)}\n")
    ids = fetch_many(names, refresh=args.refresh, min_interval=args.interval)
    print(f"\n{len(ids)} artiste(s) en cache.")

    if do_rebuild:
        print()
        store = CsvStore()
        rebuild_artists_csv(store)               # merge=True : ne touche pas aux autres artistes
        if args.rebuild_genres:
            rebuild_genres_csv(store)
    else:
        print("(CSV du simulateur non régénérés — utiliser --rebuild)")


if __name__ == "__main__":
    main()
