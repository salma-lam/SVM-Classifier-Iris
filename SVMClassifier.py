                         #TP SVM

            ####  Partie III: SVM Multi-classe  ####

#III-1-
from sklearn import datasets
iris = datasets.load_iris()


#III-2-
import numpy as np
# Copier les observations dans une matrice Data
Data = iris.data
# Copier les étiquettes dans un vecteur Label
Label = iris.target


#III-3-
from sklearn.model_selection import train_test_split
# Diviser le dataset et les labels en données d'apprentissage et de test
Data_train, Data_test, Label_train, Label_test = train_test_split(Data, Label, test_size=0.33, random_state=42)


#III-4-
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
# Créer une instance de SVM avec les paramètres souhaités
svm_classifier = SVC(kernel='linear', decision_function_shape='ovo')
# Créer un classificateur un contre un en utilisant SVM
ovo_classifier = OneVsOneClassifier(svm_classifier)
# Entraîner le modèle sur les données d'apprentissage
ovo_classifier.fit(Data_train, Label_train)


#III-5-
from sklearn.metrics import accuracy_score
# Prédire les étiquettes des données de test
predictions = ovo_classifier.predict(Data_test)
# Évaluer les performances du modèle
accuracy = accuracy_score(Label_test, predictions)
print("Accuracy:", accuracy)


#III-6-
#a-
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, precision_score

# Charger les données iris
iris = load_iris()
X = iris.data
y = iris.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Définir les différents noyaux à utiliser
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Initialiser une liste pour stocker les performances
performances = []

# Boucler sur chaque noyau
for kernel in kernels:
    # Créer le modèle SVM avec le noyau actuel
    model = SVC(kernel=kernel)
    
    # Créer un classificateur One-vs-One en utilisant SVM
    ovo_classifier = OneVsOneClassifier(model)
    
    # Entraîner le modèle
    ovo_classifier.fit(X_train, y_train)
    
    # Prédire sur l'ensemble de test
    y_pred = ovo_classifier.predict(X_test)
    
    # Calculer l'accuracy et la précision
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    
    # Déterminer la complexité de calcul en fonction du noyau
    if kernel == 'linear':
        complexity = 'Low'
    elif kernel == 'rbf':
        complexity = 'Medium'
    else:
        complexity = 'High'
    
    # Stocker les performances dans la liste
    performances.append((kernel, acc, precision, complexity))

# Afficher les performances dans le tableau
print("Tableau des performances du SVM multi-classe, un contre un:")
print("Noyaux\tAccuracy\tPrécision\tComplexité de calcul")
for performance in performances:
    print(f"{performance[0]}\t{performance[1]:.3f}\t\t{performance[2]:.3f}\t\t{performance[3]}")


#III-7-

#Question 4 : Construire le modèle de classification du SVM multi-classe
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# Créer une instance de SVM avec les paramètres souhaités
svm_classifier = SVC(kernel='linear')
# Créer un classificateur un contre tous en utilisant SVM
ovr_classifier = OneVsRestClassifier(svm_classifier)
# Entraîner le modèle sur les données d'apprentissage
ovr_classifier.fit(Data_train, Label_train)


#Question 5 : Tester le modèle sur le 1/3 de Dataset
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# Créer une instance de SVM avec les paramètres souhaités
svm_classifier = SVC(kernel='linear')

# Créer un classificateur un contre tous en utilisant SVM
ovr_classifier = OneVsRestClassifier(svm_classifier)

# Entraîner le modèle sur les données d'apprentissage
ovr_classifier.fit(Data_train, Label_train)

# Prédire les étiquettes des données de test
predictions = ovr_classifier.predict(Data_test)

# Évaluer les performances du modèle
accuracy = accuracy_score(Label_test, predictions)
print("Accuracy:", accuracy)


#Question 6 : Calcul de la performance du modèle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

# Charger les données iris
iris = load_iris()
X = iris.data
y = iris.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Définir les différents noyaux à utiliser
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Initialiser une liste pour stocker les performances
performances = []

# Boucler sur chaque noyau
for kernel in kernels:
    # Créer le modèle SVM avec le noyau actuel
    model = SVC(kernel=kernel)
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Calculer l'accuracy et la précision
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    
    # Déterminer la complexité de calcul en fonction du noyau
    if kernel == 'linear':
        complexity = 'Low'
    elif kernel == 'rbf':
        complexity = 'Medium'
    else:
        complexity = 'High'
    
    # Stocker les performances dans la liste
    performances.append((kernel, acc, precision, complexity))

# Afficher les performances dans le tableau
print("Tableau des performances du SVM multi-classe, un contre tous:")
print("Noyaux\tAccuracy\tPrécision\tComplexité de calcul")
for performance in performances:
    print(f"{performance[0]}\t{performance[1]:.3f}\t\t{performance[2]:.3f}\t\t{performance[3]}")




#III-8-

# 1. Accuracy :
#    - Les deux méthodes (un contre un et un contre tous) fournissent des performances presque parfaites (Accuracy proche de 1.000) pour les noyaux linéaire (Linear) et radial (RBF).
#    - Pour les noyaux polynomial (Poly) et sigmoid (Sigmoid), l'approche un contre tous semble mieux généraliser les données dans ce cas particulier.

# 2. Précision :
#    - La précision est élevée pour les noyaux linéaire et radial dans les deux approches.
#    - Pour le noyau polynomial, la précision est légèrement plus basse avec l'approche un contre un.
#    - La précision est très basse pour le noyau sigmoid dans les deux approches.

# 3. Complexité de calcul :
#    - La complexité de calcul est généralement plus élevée pour les noyaux non linéaires (Poly et Sigmoid), ce qui est cohérent avec la nature des SVM.

# En résumé, les deux méthodes offrent généralement de bonnes performances, mais l'approche un contre tous semble mieux fonctionner pour les noyaux polynomial et sigmoid dans ce cas spécifique.
