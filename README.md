# Atelier Scikit-learn — Prédiction de l'état des capteurs IoT

Modèle de classification (KNN) permettant de prédire automatiquement l'état d'un capteur IoT
(`OK`, `ALERTE`, `ERREUR`) à partir de ses mesures : température, humidité, pression, consommation.

Projet réalisé dans le cadre du programme **P1 Intelligence Artificielle — Orange Digital Center**.

## Structure du projet

```
atelier_scikit-learn_iot/
├── data/
│   └── mesures_capteurs.csv          # Données brutes fournies
├── notebooks/
│   └── atelier_scikit-learn_iot.ipynb  # Notebook complet du workflow ML
├── models/
│   ├── modele_capteurs.joblib        # Modèle entraîné (format Joblib)
│   └── modele_capteurs.pkl           # Modèle entraîné (format Pickle)
└── README.md
```

## Workflow suivi

```
Dataset → Chargement → Exploration → Nettoyage → X / y → Train / Test
→ Prétraitement → Modèle → fit() → predict() → Évaluation
→ Sauvegarde → Chargement → Réutilisation
```

| Étape | Détail |
|---|---|
| Nettoyage | Suppression des doublons, retrait des lignes sans cible connue |
| X / y | `X` = température, humidité, pression, consommation — `y` = état (classification multi-classes) |
| Train / Test | Split 80/20, stratifié sur `y`, `random_state=42` |
| Prétraitement | Imputation par la médiane (`SimpleImputer`) + standardisation (`StandardScaler`) |
| Modèle | `KNeighborsClassifier` (k=5) |
| Évaluation | Accuracy, matrice de confusion, rapport de classification (precision / recall / F1) |
| Bonus | Recherche du meilleur k par validation croisée (F1-score macro) |

## Principaux résultats

- Le dataset cible est **fortement déséquilibré** (≈94 % `OK`, ≈5 % `ALERTE`, ≈1 % `ERREUR`), ce qui
  rend l'accuracy seule trompeuse : un modèle prédisant toujours `OK` obtiendrait déjà un score élevé.
- Le **recall** sur les classes `ALERTE`/`ERREUR` est la métrique la plus pertinente dans ce contexte
  de maintenance IoT : mieux vaut quelques fausses alertes que rater une vraie panne.
- Le modèle est sauvegardé et rechargé avec succès pour prédire l'état d'une nouvelle mesure.

## Prérequis

- Python ≥ 3.10
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `jupyter`

## Installation et exécution

```bash
git clone <url-du-depot>
cd atelier_scikit-learn_iot

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

cd notebooks
jupyter notebook atelier_scikit-learn_iot.ipynb
```

Exécuter le notebook dans l'ordre (**Kernel → Restart & Run All**) pour reproduire l'ensemble du
workflow, y compris la régénération des modèles dans `models/`.
