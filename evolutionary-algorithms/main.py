import csv
import random
import copy


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
    num_pop = 100
    num_elite = 5
    num_mutations = 0.5 * num_pop # 50% of population gets mutated
    
    # Load in all the data
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
    for _ in range(0, 1):
        # Calculate total affinity of each individual in the population
        rankingList = []
        for individual in population:
            rankingList.append((calculateWorth(individual, knightsDict), individual))    
            
        rankingList.sort(key=lambda x:x[0], reverse=True)
        affinities.append(rankingList[0])
        
        population = [item[1] for item in rankingList[:num_elite]]
        print(population)
        
        for _ in range(0, num_mutations):
            population.append()
        
        
        
    


if __name__ == "__main__":
    main()