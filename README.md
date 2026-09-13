# autonomous-dynamic-pricing

Revenue Optimization Through Dynamic Pricing: Creating an Autonomous Pricing Agent
Using Reinforcement Learning — Travail de Bachelor (HEG Arc).

Le prix fixe unique (norme actuelle de la billetterie de spectacle vivant) est
structurellement sous-optimal : la demande varie trop d'un évènement à l'autre
et pendant la vente pour qu'un seul prix convienne. Ce projet construit un
**simulateur de marché de billetterie** (modèle à base d'agents, Mesa — pas de
données de vente réelles disponibles) calibré sur des données réelles
(ChartMetric pour la popularité des artistes, OFS pour la démographie), puis y
entraîne et compare des agents de **tarification dynamique** par apprentissage
par renforcement (Q-table Monte-Carlo, puis DQN) contre le prix fixe. Résultat :
les deux battent le prix fixe réaliste sur le chiffre d'affaires (Q-table +7 %,
DQN +14 %) sans sacrifier le taux de remplissage — détails dans le rapport.

## Structure

- `env/` — simulateur (modèle d'achat, marché, environnement Gymnasium)
- `pricing_agent/` — agents (prix fixe, Q-table, DQN) + comparaison appariée
- `scripts/` — récupération de données, entraînement, comparaison (CLI)
- `notebooks/` — pipeline de données, entraînement, comparaison, et un
  parcours de démonstration guidé (`demo_1` à `demo_4`)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
pip install -r requirements.txt
pip install -r requirements-drl.txt   # optionnel : DQN (TensorFlow)
```

## Faire tourner le projet

`data/upsert/artists.csv` et les modèles entraînés (`models/pricingAgents/`)
ne sont **pas dans ce dépôt** : ils dérivent de l'API ChartMetric, dont les
conditions d'utilisation interdisent la redistribution — incompatible avec la
licence MIT publique (détails : `models/pricingAgents/README.md`). Deux options :

1. **Les obtenir hors GitHub** et les copier aux emplacements ci-dessus, puis :
   ```bash
   python scripts/smoke_test.py                              # vérifie que l'environnement tourne
   python scripts/run_comparison.py --dqn --episodes 500     # prix fixe vs Q-table vs DQN
   ```
2. **Les régénérer soi-même**, avec un identifiant ChartMetric personnel :
   copier `.env.example` en `.env` à la racine du projet, renseigner
   `CHARTMETRIC_REFRESH_TOKEN`, puis :
   ```bash
   python scripts/fetch_chartmetric.py --rebuild
   python scripts/train_qtable.py --episodes 60000    # ~45 min CPU
   python scripts/train_dqn.py    --episodes 15000    # ~1 h CPU
   python scripts/run_comparison.py --dqn --episodes 500
   ```

Chaque entraînement sauvegarde un modèle horodaté dans `models/pricingAgents/` ;
les agents chargent toujours la version la plus récente.

## Notebooks

Pipeline de données (`01`–`06`), entraînement/comparaison (`07`–`10`), et le
parcours de démonstration (`demo_1`–`demo_4`, pensé pour être exécuté en
séquence). Détail de chacun dans sa première cellule.

⚠️ Après une modification du code de `env/`, toujours **Kernel > Restart**
avant de relancer un notebook — `%autoreload` ne recharge pas fiablement la
chaîne de dépendances `refdata → agents → market → gym_wrapper`.

## Copyright — données de l'OFS

Les fichiers suivants de `data/raw/` sont des statistiques publiques de l'**Office fédéral de la statistique (OFS)**:

| Fichier | Statistique OFS |
|---|---|
| `px-x-0304010000_205.px` | [Enquête suisse sur la structure des salaires (ESS)](https://www.bfs.admin.ch/bfs/fr/home/statistiques/travail-remuneration/enquetes/ess.html) — salaire médian |
| `participationDemographics.csv` | [Statistique des pratiques culturelles 2019](https://www.bfs.admin.ch/bfs/fr/home/statistiques/culture-medias-societe-information-sport/culture/pratiques-culturelles.html) — participation aux concerts |
| `budgetRatioDemographics.csv` | [Enquête sur le budget des ménages 2018–2019](https://www.bfs.admin.ch/news/fr/2019-0623) — dépenses « théâtre et concerts » |
> Office fédéral de la statistique (OFS).