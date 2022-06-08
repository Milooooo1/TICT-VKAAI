import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import random
import copy
import argparse

def mutate(table):
    new_table = copy.deepcopy(table)
    for _ in range(1):
        random_index1 = random.randint(0, len(new_table)-1)
        random_index2 = random.randint(0, len(new_table)-1)
        while random_index2 == random_index1:
            random_index2 = random.randint(0, len(new_table)-1)

        saveSeat = new_table[random_index1]
        new_table[random_index1] = new_table[random_index2]
        new_table[random_index2] = saveSeat
    return new_table

def crossover(eliteParent, randomParent):
    start = random.randint(2, len(eliteParent)-6)
    end = random.randint(4, len(eliteParent)-4)
    while end < start:
        end = random.randint(4, len(eliteParent)-4)
    
    child = eliteParent[start:end]
    for knight in randomParent:
        if knight not in child:
            child.append(knight)
    
    return child

def calculateWorth(population, knightsDict):
    totalAffinity = 0
    for index, knight in enumerate(population):
        l_neighbor = population[index - 1]
        if index == len(population)-1:
            r_neighbor = population[0]
        else: 
            r_neighbor = population[index + 1]
        
        l_aff = knightsDict[knight][l_neighbor] * knightsDict[l_neighbor][knight]
        r_aff = knightsDict[knight][r_neighbor] * knightsDict[r_neighbor][knight]
        totalAffinity += (l_aff + r_aff)
    return totalAffinity

def main():
    random.seed(0)
    
    knightsDict = {}
    names = []
    epochs = 500
    num_pop = 100
    num_elite = int(0.1 * num_pop)
    num_randoms = int(0.2 * num_pop)
    num_mutations = int(0.3 * num_pop)
    num_crossovers = int(0.4 * num_pop)
    
    # Load all the data
    with open('RondeTafel.csv', 'r', newline='') as csvfile:
        csv_data = csv.reader(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
       
        for index, row in enumerate(csv_data):
            if index <= 1: 
                names = row[1:] 
                continue
            knightsDict[row[0]] = dict([(names[name_index], float(item)) for name_index, item in enumerate(row[1:])])
 
    # Initial starting population
    population = []
    for _ in range(0, num_pop):
        individual = copy.deepcopy(names)
        random.shuffle(individual)
        population.append(individual)
     
    affinities = [] # best affinities of each generation
    for _ in tqdm(range(0, epochs)):
        # Calculate total affinity of each individual in the population
        rankingList = []
        for individual in population:
            rankingList.append((calculateWorth(individual, knightsDict), individual))    
            
        rankingList.sort(key=lambda x:x[0], reverse=True)
        affinities.append(rankingList[0])
        
        # Add the best elitists to the new population
        population = [item[1] for item in rankingList[:num_elite]]
        
        # Add random not elites
        for _ in range(0, num_randoms):
            population.append(random.choice(rankingList[num_elite:])[1])
 
        # Mutate the elitists and add them to the population
        for _ in range(0, num_mutations):
            population.append(mutate(random.choice(rankingList)[1]))
        
        # Cross over the elitists with randoms
        for _ in range(0, num_crossovers):
            eliteParent = random.choice(rankingList[:num_elite])[1]
            randomParent = random.choice(rankingList[num_elite:])[1]
        
            population.append(crossover(eliteParent, randomParent))
        
        # [print(pop) for pop in population]
        # print(len(population))

    print("Creating plot...")

    plt.plot(range(0, epochs), [aff[0] for aff in affinities])
    plt.title("Screeplot")
    plt.xlabel("Iteration")
    plt.ylabel("Max affinity")
    plt.savefig("screeplot.png")    

    print("Plot saved")

    affinities.sort(key=lambda x: x[0])
    print(f"Best affinity = {affinities[-1]}")
    


if __name__ == "__main__":
    main()