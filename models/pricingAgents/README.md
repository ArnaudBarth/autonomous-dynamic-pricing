# `models/pricingAgents/`

Ce dossier reçoit les modèles entraînés (`q_table_AAAAMMJJ-HHMMSS.npy`,
`dqn_AAAAMMJJ-HHMMSS.keras`, + sidecars `.meta.json`) — **volontairement vides
sur GitHub**.

## Pourquoi

Ces modèles sont entraînés sur `data/upsert/artists.csv`, lui-même dérivé de
l'API ChartMetric. Comme les conditions d'utilisation de ChartMetric
n'autorisent pas la redistribution de ses données (même dérivées), ce dépôt
étant public sous licence MIT, ni `artists.csv` ni les modèles qui en
dépendent ne sont commités (cf. `.gitignore` et le `README.md` racine,
section *Configuration des identifiants*).

## Régénérer un modèle localement

```bash
# 1. Récupérer/reconstituer data/upsert/artists.csv (nécessite un identifiant
#    ChartMetric — cf. README racine) :
python scripts/fetch_chartmetric.py --rebuild

# 2. Entraîner :
python scripts/train_qtable.py --episodes 60000
python scripts/train_dqn.py    --episodes 15000
```

Les fichiers produits ici restent locaux (ignorés par git) — c'est le
comportement voulu.
