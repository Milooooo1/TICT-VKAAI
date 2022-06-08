# Evolutionary Algorithms

## Opdracht 1: Voorbeschouwing

a. bepaal het phenotype voor dit vraagstuk. Beargumenteer je keuze.

De tafel layout is het phenotype

b. Bepaal een geschikt genotype voor dit vraagstuk. Beargumenteer je keuze.

['Galahad', 'Tristan', 'Bors the Younger', 'Arthur', 'Gawain', 'Bedivere', 'Lamorak', 'Geraint', 'Percival', 'Kay Sir Gareth', 'Lancelot','Gaheris'] dit is de orde van de tafel

c. Bepaal een geschikte fitness functie. Beargumenteer je keuze.

De fitness word bepaalt door de affiniteit van alle mensen rond de tafel naar elkaar. Dus ridder B zit naast A en C. Ridder A vind iets van B maar B ook van A, dit getal kan verschillen dus wordt de fitness berekent door naar de affiniteit van beide kanten keer elkaar.

d. Bedenkt geschikte crossover operator(en). Beargumenteer je keuze(s).

Je pakt een subsectie van de orde van A en vult die aan met mensen van B. Dit is een position based crossover functie. Met deze crossover behoud je een deel van de tafel orde van A (de elite) en geef je de rest een kans om betere ordes te verzinnen.

e. Bedenk geschikte mutatie operator(en). Beargumenteer je keuze(s).

Een geschikte mutatie is een random knight om de tafel te ruilen met een andere knight aan de tafel

## Opdracht 2: Coderen

Zie main.py hier staan ook de tweak parameters in

## Opdracht 3: Nabeschouwing

Q: "_Zou gradient descent ook een geschikte methode zijn om dit vraagstuk op te lossen? Waarom wel/niet ?_"

A: Nee want het probleem is niet makkelijk wiskundig te definieren.
