# SVM Multi-Classe avec Scikit-learn

Cet exemple met en œuvre des classificateurs multi-classes utilisant des Machines à Vecteurs de Support (SVM) avec les approches "One-vs-One" (Un contre Un) et "One-vs-Rest" (Un contre Tous), appliqués sur le dataset **Iris**.

## Contenu

- **III-1 :** Chargement des données Iris depuis la bibliothèque Scikit-learn.
- **III-2 :** Séparation des observations (features) et des étiquettes (labels).
- **III-3 :** Division du dataset en ensembles d'entraînement et de test.
- **III-4 :** Entraînement d'un modèle SVM multi-classe avec l'approche One-vs-One.
- **III-5 :** Prédiction et évaluation des performances du modèle avec la précision.
- **III-6 :** Comparaison des performances pour différents noyaux (linear, rbf, poly, sigmoid) dans l'approche One-vs-One.
- **III-7 :** Construction d'un modèle SVM multi-classe avec l'approche One-vs-Rest.
- **III-8 :** Comparaison des performances pour différents noyaux dans l'approche One-vs-Rest.

## Utilisation

### Installation des bibliothèques nécessaires

Avant de pouvoir exécuter le code, installez les bibliothèques requises avec :

```bash
pip install scikit-learn numpy
