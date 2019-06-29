# Put the imports you need here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import csv
from pprint import pprint as print # pretty printing, easier to read but takes more room

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        #SUGGESTION - What if we wanted to use a different distance
        # metric? Would that make sense for this problem?
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = None
        self.fitness = None
    
    def routeDistance(self):
        if self.distance == None:
            pathDistance = 0.0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i+1 < len(self.route):
                    toCity = self.route[i+1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == None:
            self.fitness = 1 / float(self.routeDistance())
            #SUGGESTION - Is the scaling an issue with this method
            # of defining fitness? Would negative distance make more
            # sense (obviously with properly defined selection functions)
        return self.fitness
    
def genCityList(filename):
    cityList = []
    #TODO - implement this function by replacing the code between the TODO lines
    '''
    for i in range(0,12):
        cityList.append(City(x=int(random.random() * 200),
                             y=int(random.random() * 200)))
    '''
    data = pd.read_csv(filename, sep=" ", header=None, names=["index", "x", "y"])
    data.set_index('index', inplace=True)
    #print(data)
    
    for index, row in data.iterrows():
        #access data using column names
        #print(str(row['x']) + " " + str(row['y']))
        cityList.append(City(row['x'], row['y']))
    
    #TODO - the code above just generates 12 cities (useful for testing)
    return cityList

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    #SUGGESTION - Could population be 'seeded' with known good routes?
    # In other words, would heuristic initialization help?
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    #return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    # lambda x : x[1] will return the value of an item in the dict
    return sorted(fitnessResults.items(), key = lambda x : x[1], reverse = True)###Possible Error

def parentSelection(population, popRanked, poolSize=None):
    """
    Note that this function returns only poolSize City instances. This
    is useful if we are doing survivorSelection as well, otherwise we
    can just set poolSize = len(population).
    """
    
    if poolSize == None:
        poolSize = len(popRanked)
    
    matingPool = []
    
    #TODO - implement this function by replacing the code between the TODO lines
    '''
    for i in range(0, poolSize):
        fitness = Fitness(population[i]).routeFitness()
        matingPool.append(random.choice(population))
    '''
    ###1st approach - Fitness proportionate selection ###
    df = pd.DataFrame(np.array(popRanked), columns=['Index', "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()# calculate the cumulative sum
    df['cum_perc'] = 100 * df.cum_sum/df.Fitness.sum()# convert the cum_sum to percentage form
    selectionResults = []
    #print(df)# to be removed
    
    for i in range(0, poolSize):
        randPerc = 100 * random.random()# generate random percentage
        for i in range(0, len(popRanked)):
            # compare the randPerc with generated percentage list
            # if the randPerc is within a paricular chromosome's cum_perc then the chromosome will be selected and
            #therefore, the chromosome with higher cum_perc i.e. fitness will have a greater chance to be selected
            # Note that the chromosome;s index instead of the chromosome will be stored
            if randPerc <= df.iat[i,3]:###Possible Error
                selectionResults.append(popRanked[i][0])
                break
    
    #TODO - the code above just randomly selects a parent. Replace
    # it with code which implements one of the parent selection
    # strategies mentioned in the lecture.
    for i in range(0, len(selectionResults)):
        matingPool.append(population[selectionResults[i]])
        #print(selectionResults[i])# to be removed
    
    return matingPool

def survivorSelection(population, popRanked, eliteSize):
    """
    This function returns a list of length eliteSize (the selected
    City instances which will be preserved)
    """
    
    elites = []
    selectionResults = []
    
    #TODO - implement this function by replacing the code between the TODO lines
    for i in range(eliteSize):
        #selectionResults.append(popRanked[-(i + 1)][0])
        selectionResults.append(popRanked[i][0])
    #TODO - the code above just selects the first eliteSize City instances.
    # Replace it with code which selects the best individuals
    #SUGGESTION - age-based survivor selection isn't trivial to implement
    # based on this notebook, as you would need to make changes to how
    # the chromosomes are stored. Consider it a fun challenge (not
    # required, no bonus marks) for those who find this lab too easy.
    for i in range(0, len(selectionResults)):
        elites.append(population[selectionResults[i]])
    
    return elites

def crossover(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene + 1):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child
    
    #TODO - the code above simply generates new random routes.
    # Replace it with code which implements a suitable crossover method.

def breedPopulation(matingpool, poolSize):
    children = []
    
    for i in range(1, len(matingpool), 2):
        child1 = crossover(matingpool[i-1], matingpool[i])
        child2 = crossover(matingpool[i], matingpool[i-1])
        children.append(child1)
        children.append(child2)
    
    '''
    pool = random.sample(matingpool, len(matingpool))
    
    for i in range(0, poolSize):
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    '''
    #SUGGESTION - would randomly choosing parents from matingpool make
    # a difference compared to just choosing them in order? Wouldn't be
    # too hard to test that, would it?
    
    return children

def mutate(route, mutationProbability):
    """
    mutationProbability is the probability that any one City instance
    will undergo mutation
    """
    mutated_route = route[:]
    for swapped in range(len(route)):
        if (random.random() < mutationProbability):
            #TODO - implement this function by replacing the code between
            # the TODO lines
            '''
            city1 = route[i]
            city2 = route[i-1]
            mutated_route[i] = city2
            mutated_route[i-1] = city1
            '''
            ###1st approach - swap mutation###
            swapWith = random.randint(0, len(route) - 1)
            
            city1 = route[swapped]
            city2 = route[swapWith]
            mutated_route[swapped] = city2
            mutated_route[swapWith] = city1
            
            #TODO - the code above simply swaps a city with the city
            # before it. This isn't really a good idea, replace it with
            # code which implements a better mutation method
    
    return mutated_route

def mutation(population, mutationProbability):
    mutatedPopulation = []
    for i in range(0, len(population)):
        mutatedIndividual = mutate(population[i], mutationProbability)
        mutatedPopulation.append(mutatedIndividual)
    return mutatedPopulation

def oneGeneration(population, eliteSize, mutationProbability):
    
    # First rank the chromosomes in the population
    popRanked = rankRoutes(population)
    
    # First we preserve the elites
    elites = survivorSelection(population, popRanked, eliteSize)
    
    # Then we calculate what our mating pool size should be and generate
    # the mating pool
    poolSize = len(population) - eliteSize
    matingpool = parentSelection(population, popRanked, poolSize)
    #SUGGESTION - What if the elites were removed from the mating pool?
    # Would that help or hurt the genetic algorithm? How would that affect
    # diversity? How would that affect performance/convergence?

    # Then we perform crossover on the mating pool
    children = breedPopulation(matingpool, poolSize)
    
    # We combine the elites and children into one population
    new_population = elites + children
    
    # We mutate the population
    mutated_population = mutation(new_population, mutationProbability)
    #SUGGESTION - If we do mutation before selection and breeding, does
    # it make any difference?
    
    return mutated_population


start_time = time.time()
'''
filename = 'TSPdata/tsp-case04.txt'
popSize = 40
eliteSize = 10
mutationProbability = 0.015
iteration_limit = 250
'''
filename = 'TSPdata/tsp-case03.txt'
popSize = 100
eliteSize = 20
mutationProbability = 0.025
iteration_limit = 1500

cityList = genCityList(filename)

population = initialPopulation(popSize, cityList)
distances = [Fitness(p).routeDistance() for p in population]
min_dist = min(distances)
print("Best distance for initial population: " + str(min_dist))
progress = []
progress.append(1 / rankRoutes(population)[0][1])

for i in range(iteration_limit):
    population = oneGeneration(population, eliteSize, mutationProbability)
    distances = [Fitness(p).routeDistance() for p in population]
    min_dist = min(distances)
    print("Best distance for population in iteration " + str(i) +
          ": " + str(min_dist))
    progress.append(1 / rankRoutes(population)[0][1])

plt.plot(progress)
plt.ylabel('Distance')
plt.xlabel('Generation')
plt.show()
    #TODO - Perhaps we should save the best distance (or the route itself)
    # for plotting? A plot may be better at demonstrating performance over
    # iterations.
    #SUGGESTION - You could also print/plot the N best routes per
    # iteration, would this give more insight into what's happening?
    #SUGGESTION - The suggested code in this cell stops when a specific
    # number of iterations are reached. Would it help to implement
    # a different stopping criterion (e.g. best fitness no longer
    # improving)?

end_time = time.time()
print("Time taken: {} s".format(end_time-start_time))

filename = 'mysolution.csv'
distances = [Fitness(p).routeDistance() for p in population]
index = np.argmin(distances)
best_route = population[index]
with open(filename, mode='w') as f:
    writer = csv.writer(f, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(best_route)):
        writer.writerow([i, best_route[i].x, best_route[i].y])