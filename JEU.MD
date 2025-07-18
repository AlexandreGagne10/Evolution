# Évolution: Simulation d'Adaptation

Ce projet propose une simulation visant à étudier des comportements d'évolution dans un environnement simple. Le joueur peut modifier différents paramètres via un panneau de contrôle pour observer l'impact sur la population d'agents (particules) au fil des rondes.

## Description générale

* La carte est une zone carrée comportant un périmètre qui sert de zone de repos pour les agents.
* De la nourriture apparaît aléatoirement sur la carte à chaque ronde.
* Une journée correspond à **une ronde** de simulation.
* Les agents possèdent une espérance de vie fixée à `X` jours/ronde.
* Pour survivre à la ronde suivante, chaque agent doit consommer `X` unités de nourriture. Faute de quoi, il meurt.
* Les agents peuvent se reproduire lorsqu'ils ont accumulé suffisamment d'énergie (nourriture) et qu'ils rencontrent un partenaire.
* Les paramètres (quantité de nourriture générée, durée de vie, coûts de reproduction, etc.) sont ajustables via le panneau de contrôle.

## Règles détaillées

1. **Génération de nourriture**
   - À chaque ronde, une quantité déterminée (et configurable) de nourriture est générée à des positions aléatoires sur la carte.
   - Si plusieurs agents tentent de prendre la même portion de nourriture, seul le premier arrivé la consomme.

2. **Déplacement et repos**
   - Les agents se déplacent librement sur la carte pendant une ronde.
   - S'ils atteignent le périmètre extérieur, ils entrent dans la zone de repos et ne peuvent plus se déplacer pour cette ronde.

3. **Consommation et survie**
   - Chaque agent doit avoir accumulé au moins `X` unités de nourriture pour rester en vie à la fin de la ronde.
   - Les unités de nourriture consommées au-delà de ce seuil peuvent être stockées pour la reproduction.

4. **Reproduction**
   - Deux agents disposant d'une quantité de nourriture suffisante peuvent créer un "bébé". Les ressources nécessaires sont prélevées sur leurs réserves.
   - Les caractéristiques des agents parents (longévité, vitesse, etc.) peuvent être transmises avec une légère mutation pour simuler l'évolution.

5. **Fin de vie**
   - Lorsqu'un agent atteint la limite de son espérance de vie ou n'a pas mangé suffisamment, il disparaît de la simulation.

## Observations attendues

Cette simulation permet d'observer différents phénomènes :

* L'adaptation des agents en fonction de la quantité de nourriture disponible.
* Les stratégies de déplacement ou de repos influençant la survie et la reproduction.
* L'émergence possible de comportements collectifs (coopération ou compétition) selon les paramètres choisis.

En modifiant les règles de base et la génération aléatoire de nourriture, on peut expérimenter diverses stratégies d'évolution et analyser leur efficacité au fil du temps.

