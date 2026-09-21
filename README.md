# CarPredictor — Estimation du prix de revente de véhicules d'occasion

CarPredictor est un système de Machine Learning qui estime le prix de revente d'une voiture d'occasion à partir de ses caractéristiques principales. Il a été conçu pour une plateforme de vente de véhicules d'occasion avec trois objectifs :

- **anticiper** le prix de vente d'un véhicule avant sa mise en ligne ;
- **améliorer la transparence** entre acheteurs et vendeurs ;
- **conseiller** les utilisateurs sur la valorisation de leur véhicule.

---

## Sommaire

1. [Jeu de données](#jeu-de-données)
2. [Structure du projet](#structure-du-projet)
3. [Installation](#installation)
4. [Exécution](#exécution)
5. [Démarche](#démarche)
6. [Résultats](#résultats)
7. [Interface de prédiction](#interface-de-prédiction)
8. [Gestion de projet](#gestion-de-projet)

---

## Jeu de données

Les données proviennent d'un historique d'annonces de véhicules. Le nombre de variables est volontairement limité afin de faciliter la sélection des variables pertinentes.

| Colonne         | Type        | Description                                           |
|-----------------|-------------|-------------------------------------------------------|
| `Year`          | numérique   | Année de mise en circulation                          |
| `Km_Driven`     | numérique   | Kilométrage parcouru                                  |
| `Fuel`          | catégoriel  | Type de carburant (Petrol, Diesel, CNG, LPG, Electric)|
| `Seller_Type`   | catégoriel  | Type de vendeur (Individual, Dealer, …)               |
| `Transmission`  | catégoriel  | Type de boîte (Manual, Automatic)                     |
| `Owner`         | ordinal     | Nombre de propriétaires précédents                    |
| `Selling_Price` | numérique   | **Variable cible** — prix de vente                    |

Le fichier brut doit être placé dans `data/raw/`.

---

## Structure du projet

```
CarPredictor/
├── data/
│   ├── raw/                  # Données brutes (non modifiées)
│   └── processed/            # Données nettoyées et prêtes pour l'entraînement
├── notebooks/
│   ├── 01_exploration_nettoyage.ipynb   # Étape 1 — EDA & nettoyage
│   ├── 02_modeles_baseline.ipynb        # Étape 2 — Modèles par défaut
│   ├── 03_optimisation.ipynb            # Étape 3 — GridSearch / RandomizedSearch
│   └── 04_comparaison_selection.ipynb   # Étape 4 — Comparaison & choix final
├── src/
│   ├── preprocessing.py      # Nettoyage, encodage, pipeline de prétraitement
│   ├── train.py              # Entraînement et évaluation des modèles
│   └── predict.py            # Chargement du modèle et prédiction
├── models/
│   └── best_model.joblib     # Modèle final exporté (pipeline complet)
├── app/
│   └── app.py                # Interface de saisie et d'estimation du prix
├── reports/
│   └── figures/              # Graphiques générés (distributions, résidus, …)
├── requirements.txt
└── README.md
```

---

## Installation

Prérequis : **Python 3.10+**

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd CarPredictor

# 2. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

Principales bibliothèques utilisées : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`, `streamlit`.

---

## Exécution

Les étapes doivent être exécutées dans l'ordre, chacune s'appuyant sur la précédente.

```bash
# Étapes 1 à 4 : exploration, entraînement, optimisation, sélection
jupyter notebook notebooks/

# Ou en script : entraîne les modèles et exporte le meilleur dans models/
python src/train.py

# Étape 5 : lancer l'interface de prédiction
streamlit run app/app.py
```

---

## Démarche

### Étape 1 — Exploration & nettoyage des données

- Chargement avec Pandas, vérification des types et des dimensions.
- **Analyse exploratoire** : statistiques descriptives (moyenne, médiane, écart-type, fréquences), histogrammes du kilométrage et du prix, matrice de corrélation, heatmap et pairplot pour étudier l'effet de l'année et du kilométrage sur le prix.
- **Sélection des variables** : justification de la pertinence de chaque colonne pour la prédiction.
- **Valeurs manquantes** : imputation par la médiane (numériques) et par le mode (catégorielles).
- **Doublons** : détection et suppression.
- **Valeurs aberrantes** : détection par boîtes à moustaches, z-score (> 3) et IQR sur `Selling_Price`, `Km_Driven` et `Year`. Chaque valeur détectée est analysée puis conservée, corrigée ou supprimée selon qu'elle est plausible ou clairement incohérente.
- **Encodage** : one-hot encoding de `Fuel`, `Seller_Type` et `Transmission` ; encodage ordinal de `Owner`.
- **Découpage** : `train_test_split` 80 % / 20 %.
- **Mise à l'échelle** : `StandardScaler` (ou `MinMaxScaler`) sur les variables numériques.

### Étape 2 — Modèles de référence

Quatre familles d'algorithmes sont entraînées avec leurs paramètres par défaut, chacune dans un `Pipeline` Scikit-learn (prétraitement + modèle) afin d'éviter toute fuite de données :

- Régression Linéaire
- Random Forest
- XGBoost
- SVR

Évaluation sur l'ensemble de test avec **RMSE**, **MAE** et **R²**. Ces résultats servent de base de comparaison.

### Étape 3 — Optimisation des hyperparamètres

Les modèles les plus prometteurs sont optimisés avec `GridSearchCV` / `RandomizedSearchCV` et une validation croisée à 5 folds :

| Modèle        | Hyperparamètres explorés                              |
|---------------|-------------------------------------------------------|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`      |
| XGBoost       | `learning_rate`, `max_depth`, `subsample`, `n_estimators` |

Les performances sont comparées avant et après optimisation, puis les modèles optimisés sont réentraînés sur l'ensemble d'entraînement complet.

### Étape 4 — Comparaison et sélection du modèle final

- Graphiques de résidus et nuages de points prédictions vs valeurs réelles.
- Tableau récapitulatif (DataFrame) des métriques de tous les modèles.
- Choix du modèle final selon la **performance** (RMSE, MAE, R²) **et la robustesse** (variance des erreurs, stabilité en validation croisée).

### Étape 5 — Mise en situation réelle

- Export du pipeline final avec `joblib.dump()` dans `models/best_model.joblib`.
- Interface de saisie permettant d'obtenir une estimation instantanée du prix.

### Étape 6 — Documentation et reproductibilité

- Code commenté et cellules Markdown à chaque étape clé.
- Ce README.
- Planification du projet dans Jira (Epics et tickets).

---

## Résultats

*Tableau à compléter après l'exécution des étapes 2 à 4.*

| Modèle                   | RMSE | MAE | R²  |
|--------------------------|------|-----|-----|
| Régression Linéaire      |      |     |     |
| Random Forest            |      |     |     |
| XGBoost                  |      |     |     |
| SVR                      |      |     |     |
| Random Forest (optimisé) |      |     |     |
| XGBoost (optimisé)       |      |     |     |

**Modèle retenu :** *à compléter, avec la justification (performance et robustesse).*

---

## Interface de prédiction

L'interface demande les caractéristiques du véhicule :

- année de mise en circulation ;
- kilométrage ;
- type de carburant ;
- type de vendeur ;
- type de transmission ;
- nombre de propriétaires précédents.

Elle charge le modèle exporté et affiche immédiatement le prix estimé. Exemple d'utilisation en Python :

```python
import joblib
import pandas as pd

model = joblib.load("models/best_model.joblib")

voiture = pd.DataFrame([{
    "Year": 2015,
    "Km_Driven": 70000,
    "Fuel": "Diesel",
    "Seller_Type": "Individual",
    "Transmission": "Manual",
    "Owner": "First Owner",
}])

print(f"Prix estimé : {model.predict(voiture)[0]:,.0f}")
```

Le modèle exporté étant un pipeline complet, les données brutes peuvent être passées telles quelles : le prétraitement (encodage, mise à l'échelle) est appliqué automatiquement.

---

## Gestion de projet

Le projet est planifié dans **Jira**, avec un Epic par étape :

| Epic | Contenu                                   |
|------|-------------------------------------------|
| E1   | Exploration & nettoyage des données       |
| E2   | Entraînement des modèles de référence     |
| E3   | Optimisation des hyperparamètres          |
| E4   | Comparaison et sélection du modèle final  |
| E5   | Export du modèle et interface de prédiction |
| E6   | Documentation et reproductibilité         |

Chaque Epic est découpé en tickets correspondant aux tâches décrites dans la section [Démarche](#démarche).
