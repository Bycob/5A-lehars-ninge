# Travaux Pratiques d'Apprentissage Supervisé à l'Institut National des Sciences Appliquées de Toulouse.

*Important*

Ce repository contient également les TP d'apprentissage non supervisé.

Les répertoires `TP1-knn`, `TP2-ann` et `TP3-SVM` contienent les scripts python relatifs au TP d'apprentissage supervisé. Le répertoire `TP123-Apprentissage` contient le script utilisé pour faire la synthèse des 3 types de modèles.

## Travail pratique numéro 1

Exécuter à l'aide de la commande suivante (affiche les options disponibles):
```
TP1/tp1.py --help
```
Pour faire varier un paramètre et trouver le meilleur résultat:
```
TP1/tp1.py --compare ["neighbors", "split", "distance"]
```
Pour indiquer le nombre de voisin:
```
TP1/tp1.py --neighbors [int k]
```
Pour indiquer le pourcentage de donnée utilisé pour le training:
```
TP1/tp1.py --split [float percentage]
```
Exposant de la distance de Minkowski:
```
TP1/tp1.py --distance [float p]
```
Multithreading:
```
TP1/tp1.py --multithread
```
Pour voir l'image n et afficher la classe correspondante:
```
TP1/tp1.py --visualize_data [int n]
```
