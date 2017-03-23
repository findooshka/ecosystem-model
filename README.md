# Modélisation de populations des écosystèmes
----------------------------------------------------------------------------

## Description :

Le but de ce projet est de faire un modèle d'un écosystème. Dans ce modèle l'écosysteme se présente par les individus (animaux, par exemple) de espèces différentes, qui se déplacent sur un plan 2D. Chaque individu a un niveau de satiété qui constamment diminue (sauf si l'individu est une plante, dans ce cas sa satiété augmente). Ayant une certaine satiété un individu se reproduit, réduisant sa satiétié. Entre deux reproductions successives il y a un temps qui est plus ou egale à un temps minimal prédifini.

Un individu augmente sa satiété en mangeant des individus d'autres espèces. L'ensemble d'espèces qui sont mageable pour un individu est prédifini et il est commun pour les individus de même espèce.

## Semaine 1 :
*(vers 23/02/2017)*

Ce qui est déjà fait:
* class d'individu <code>Being</code>
* class contenant tout les individus <code>Map</code> avec des fonctions pour:
  - mouvement aléatoire avec un vecteur moyenne de déplacement
  - la nourriture
  - la reproduction
  - la chasse (poursuite de proie)
  - initialisation
  - itération
* animation des iteration
* graphes d'évolution de populations des espèces

![semi-log](https://github.com/findooshka/ecosystem-model/blob/master/graphics/semi_log.png)

![graphs separes](https://github.com/findooshka/ecosystem-model/blob/master/graphics/separate.png)

*(séance du 23/02/2017)*

Ce que l'ont veux faire lors de cette séance et des prochaines:
* expérience avec differents paramètres
* création de l évolution des paramètres
* amélioration de la reproduction(reproduction a deux, sexe)
* comportement de groupe(meute, troupeau)


Lors de cette séance nous avons décidé que ce soir chez nous, nous ferions tourner nos machines avec différents paramètres afin d'afficher les courbes pour trouver les meilleures combinaisons de paramètres possibles. Pour cela nous sommes en train de préparer des fonction de variation des paramètres que nous allons utiliser lors de la nuit. Ceci est le point de notre projet que nous voulons finaliser en prioritée.

## Semaine 2 :
 *(séance du 02/03/2017)*
 
 Pour cette séance/semaine nous avons :
 * ajouté une fonction qui fait les meusures (measurement.py)
 * implementé des algorithmes pour trouver des bonnes parametres initiales
 * trouvé un article qui traite le même sujet que notre projet d'ARE
 * Commentaires du code
 
On continue de faire les tests selon les paramètres afin d'affiner/trouvrer les paramètres idéal pour obtenir un ecosytème stable.


## Semaine 3 :
 *(séance du 09/03/2017)*
 
 Ce qui est ajouté :
 * Fonctionnalité de maladie
 * Fonctionnalité de mort de veilliesse 
 * Fonctionnalité de list de statistique d'une simulation
 * Commentaires du code
 

## Semaine 4 :
 *(séance du 16/03/2017)*
 
 Graphe de dépendance entre deux populations :
 
![deux_populations](https://github.com/findooshka/ecosystem-model/blob/master/graphics/2-especes-maladie--0.04-50-50.png)

 On fait des tests en variant plusieurs paramètres. On observe que les fonctionnalités de maladie et de mort de veilliesse permettent d´obtenir un système plus stable.