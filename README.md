# Rentabilité et recommandation de films

Nettoyage et analyse d'une base de métadonnées de films, prédiction de la **rentabilité** d'un film par un SVM après ACP, et **système de recommandation** de films basé sur le contenu (TF-IDF et similarité cosinus).

Projet de machine learning réalisé en groupe à l'ISSEA-CEMAC.

## Objectifs

1. Préparer une base de films brute et hétérogène pour l'analyse (doublons, valeurs aberrantes, valeurs manquantes, montants financiers de périodes différentes).
2. Comprendre ce qui est lié à la rentabilité d'un film (genre, langue, classification du public, popularité, avis).
3. Prédire si un film est **profitable** ou **déficitaire**.
4. Recommander des films proches d'un film donné.

## Données

- Base `movie_metadata` : **5 043 films**, 28 variables au départ (titre, réalisateur, acteurs, genres, mots-clés, budget, recettes, note IMDb, nombre de votes, likes, etc.).
- Après nettoyage : **4 783 films**.
- **Source des données** : [à compléter, avec le lien et la licence].

## Préparation des données

- **Doublons** : suppression des doublons parfaits, puis traitement cas par cas des films présents plusieurs fois avec des valeurs différentes (par exemple nombre de votes ou likes mis à jour), en conservant la ligne la plus récente.
- **Séries** : retrait des enregistrements correspondant à des séries (durée très courte ou longue) plutôt qu'à des films.
- **Valeurs aberrantes et extrêmes** : identification et traitement variable par variable.
- **Valeurs manquantes** : traitement par film puis par variable, avec une stratégie adaptée à chacune (année, durée, langue, pays, couleur, acteurs, mots-clés, classification, format d'image, budget et recettes).
- **Montants financiers** : **déflation** du budget et des recettes en dollars constants pour pouvoir comparer des films de périodes différentes, puis **imputation par KNN** des montants manquants.
- **Valeurs rares** : regroupement des modalités peu fréquentes (réalisateurs, acteurs, pays, langues, classifications, genres).

## Analyse descriptive

- Rentabilité (profit, retour sur investissement, marge nette, taux de succès) selon le **genre**, la **langue** et la **classification** du film.
- Matrice de corrélation, évolution par décennie, films les plus et les moins rentables.
- Acteurs et réalisateurs les plus rentables, et fonction de résumé de la carrière d'un acteur.

## Modèle de prédiction de la rentabilité

- **Cible** : `is_profitable` (film profitable ou déficitaire), construite à partir du budget et des recettes déflatés.
- **Variables explicatives (7)** : `movie_facebook_likes`, `social_media_score`, `num_voted_users`, `num_user_for_reviews`, `num_critic_for_reviews`, `imdb_score`, `age`.
- **Séparation** stratifiée : 3 826 films pour l'entraînement, 957 pour le test (80 % / 20 %).
- **ACP** ajustée sur l'entraînement : **4 axes** retenus.
- **Choix du modèle** : la normalité des axes n'étant pas vérifiée (tests de Shapiro-Wilk), le choix s'est porté sur un **SVM**.
- **Optimisation** : `GridSearchCV` avec validation croisée à 5 plis, noyau RBF, C et gamma optimisés, score F1 comme critère. Meilleurs paramètres : C = 0,1 et gamma = 1.

### Résultats (ensemble de test)

| Métrique | Valeur |
|---|---|
| Accuracy | 68,2 % |
| Précision (classe profitable) | 68,6 % |
| Rappel (classe profitable) | 60,7 % |
| F1 (classe profitable) | 64,4 % |
| AUC | 0,734 |

Le modèle classe correctement environ 68 % des films avec un équilibre raisonnable entre les deux classes. Le rappel sur les films profitables (60,7 %) montre qu'une partie des succès n'est pas détectée à partir de ces seules variables.

## Système de recommandation

- **Approche** : filtrage basé sur le contenu. Chaque film est décrit par un « sac de mots » regroupant genres, mots-clés, réalisateur et acteurs.
- **Vectorisation** : TF-IDF (unigrammes et bigrammes, mots vides anglais retirés), soit **4 783 films et 3 864 termes**.
- **Similarité** : cosinus entre films.
- **Reclassement** : les 20 films les plus proches sont repondérés par la note IMDb et le nombre de votes, puis les 5 meilleurs sont renvoyés.
- **Exemples** : recommandations pour *The Dark Knight*, *Avatar* et *Forrest Gump*. Les résultats sont cohérents sur le plan thématique (par exemple des films d'action-aventure et de science-fiction pour *Avatar*) et perfectibles pour d'autres films.

## Limites et pistes d'amélioration

- Les performances de classification restent modestes : ajouter d'autres variables (budget, genre, casting) et comparer d'autres modèles (régression logistique, forêts aléatoires, gradient boosting).
- Le moteur de recommandation est pénalisé par les jetons génériques (par exemple « unknown_actor ») : les retirer améliorerait la pertinence.
- La recommandation n'est évaluée que qualitativement sur quelques films.

## Structure du dépôt

```
movie-profitability-and-recommendation/
├── README.md
├── notebooks/
│   └── projet_ML_II.ipynb
├── data/             # non versionné (ou échantillon si la licence le permet)
└── requirements.txt
```

## Auteurs

- NGANONGO Juste Adji
- DJONA WAFO FRANK MAEL 
- KENFACK JOSEPHINE ELZA
- SOKOUTCHOP SOKOUDJOU DIVAN BRAYAN 

Projet académique, ISSEA-CEMAC.
