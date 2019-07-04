#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithm Lab

# ## Imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import csv
from pprint import pprint as print # pretty printing, easier to read but takes more room


# ## Convenience Classes
# 
# The 'City' class allows us to easily measure distance between cities. A list of cities is called a route, and will be our chromosome for this genetic algorithm.

# In[2]:


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# The 'Fitness' class helps to calculate both the distance and the fitness of a route (list of City instances).

# In[3]:


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
        return self.fitness


# ## Initialization Step
# 
# Initialization starts with a large **population** of randomly generated chromosomes. We will use 3 functions. The first one generates a list of cities from a file.

# In[4]:


def genCityList(filename):
    cityList = []
    
    #Read the data from textfile into pandas dataframe
    data = pd.read_csv(filename, sep=" ", header=None, names=["index", "x", "y"])
    data.set_index('index', inplace=True)
    
    #Iterate through the dataframe and use x and y coordinates to create City objects
    for index, row in data.iterrows():
        cityList.append(City(row['x'], row['y']))
    
    return cityList


# The second function generates a random route (chromosome) from a list of City instances.

# In[5]:


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# The third function repeatedly calls the second function to create an initial population (list of routes).

# In[6]:


def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


# ## Selection
# 
# Parent selection is the primary form of selection, and is used to create a mating pool.

# In[7]:


'''
This function is used to rank the routes in the population in descending order
This function returns the indexes of the routes rather the actual routes
''' 
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = lambda x : x[1], reverse = True)


# In[8]:


def parentSelection(population, poolSize=None):
    
    ###Tournament selection###
    if poolSize == None:
        poolSize = len(population)
    
    matingPool = []
    
    #define the size of the sample to be taken from the population
    tournament_size = 0.2 * len(population)
    
    '''
    Sample 20% of the routes from the population, rank the routes and retrieve the route with highest fitness
    Repeat the step above for the size of poolSize
    '''
    for i in range(0, poolSize):
        randPop = random.sample(population, int(tournament_size))
        best = randPop[0]
        for i in range(0, len(randPop)):
            if Fitness(randPop[i]).routeFitness() > Fitness(best).routeFitness():
                best = randPop[i]
        matingPool.append(best)
        
    return matingPool


# Another form of selection is survivor selection, which is used to ensure certain individuals (normally high fitness ones) survive to the next generation.

# In[9]:


def survivorSelection(population, popRanked, eliteSize):
    elites = []
    selectionResults = []
    
    ###Fitness based survivor selection
    '''
    Note: popRanked stores the ranked routes
    Get the indexes of the first eliteSize routes with top fitnesses
    '''
    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])
        
    #Get the actual routes using the indexes in selectionResults
    for i in range(0, len(selectionResults)):
        elites.append(population[selectionResults[i]])

    return elites


# ## Crossover
# 
# The crossover function combines two parents in such a way that their children inherit some of each parent's characteristics. In the case of TSP, you will need to use crossover methods such as Davis' Order Crossover (other examples are listed in the lecture slides).

# In[10]:


def crossover(parent1, parent2):
    ###Davis’ Order Crossover (OX1)###
    child = [None] * len(parent1)
    
    #generate a random slice within the chromosome
    gene1 = random.randint(0, len(parent1) - 1)
    gene2 = random.randint(0, len(parent1) - 1)
    
    # check for identical genes i.e. gene1 == gene2
    while gene1 == gene2:
        gene1 = random.randint(0, len(parent1) - 1)
        gene2 = random.randint(0, len(parent1) - 1)
    
    # sort the order
    startGene = min(gene1, gene2)
    endGene = max(gene1, gene2)
    
    # get the slice of the parent 1 chromosome and put into the child
    for i in range(startGene, endGene + 1):
        child[i] = parent1[i]
    
    # copy remained unused genes from second parent to the child, wrapping around the list
    count = endGene + 1#to indicate gene position in parent
    childCount = endGene + 1#to indicate gene position in child
    isComplete = False
    while not isComplete:
        if count == len(parent1):
            count = 0
        elif count == endGene:
            if None in child:# presence of None indicates the child is not fully filled up with genes yet
                if childCount == len(child):
                        childCount = child.index(None)
                if parent2[count] not in child:
                    child[childCount] = parent2[count]
            isComplete = True
        else:
            if parent2[count] not in child:# presence of None indicates the child is not fully filled up with genes yet
                if None in child:
                    if childCount == len(child):
                        childCount = child.index(None)
                    child[childCount] = parent2[count]
                    childCount += 1
            count += 1
    
    return child


# Crossover should be run on pairs from the mating pool to produce a new generation (of the same size).

# In[11]:


def breedPopulation(matingpool, poolSize):
    children = []
    
    for i in range(0, poolSize):
        child = crossover(matingpool[i-1], matingpool[i])
        children.append(child)
        
    return children


# ## Mutation
# 
# Mutations are small random changes which maintain/introduce diversity. By necessity, mutations must occur at low probability and avoid changing everything in a chromosome. As with crossover, mutation in TSP must respect the constraint that every City occurs exactly once in the Route.

# In[12]:


def mutate(route):
    ###Shuffle mutation###
    portionLen = int(0.02 * len(route))
    
    #get a random portion of a route and shuffle that portion
    idx = random.randint(0, len(route) - portionLen)
    portion = route[idx : idx + portionLen]
    route[idx : idx + portionLen] = random.sample(portion, len(portion))    
    
    return route


# The mutate function needs to be run over the entire population, obviously.

# In[13]:


def mutation(population):
    mutatedPopulation = []
    for i in range(0, len(population)):
        mutatedIndividual = mutate(population[i])
        mutatedPopulation.append(mutatedIndividual)
    return mutatedPopulation


# ## Running One Generation
# 
# Now that we have (almost) all our component functions in place, let's call them altogether.

# In[14]:


def oneGeneration(population, eliteSize):
    
    # First rank the routes i.e. chromosomes in the population
    popRanked = rankRoutes(population)
    
    # First we preserve the elites
    elites = survivorSelection(population, popRanked, eliteSize)
    
    # Then we calculate what our mating pool size should be and generate
    # the mating pool
    poolSize = len(population) - eliteSize
    matingpool = parentSelection(population, poolSize)
    
    # Then we perform crossover on the mating pool
    children = breedPopulation(matingpool, poolSize)
    
    # We combine the elites and children into one population
    new_population = elites + children
    
    # We mutate the population
    mutated_population = mutation(new_population)
    
    return mutated_population


# ## Running Genetic Algorithm
# 
# The entire genetic algorithm needs to initialize a Route of City instances, then iteratively generate new generations. Take note that, unlike all the cells above, the cell below is NOT a function. Various parameters are set right at the top (you should set them to something reasonable).

# In[17]:


start_time = time.time()
filename = 'TSPdata/tsp-case04.txt'
popSize = 65
eliteSize = 16
iteration_limit = 500
'''
filename = 'TSPdata/tsp-case04.txt'
popSize = 60
eliteSize = 15
iteration_limit = 300
'''
'''
filename = 'TSPdata/tsp-case03.txt'
popSize = 20
eliteSize = 5
iteration_limit = 100
'''
cityList = genCityList(filename)

population = initialPopulation(popSize, cityList)
distances = [Fitness(p).routeDistance() for p in population]
min_dist = min(distances)
print("Best distance for initial population: " + str(min_dist))
progress = []
progress.append(1 / rankRoutes(population)[0][1])#append the best route in each iteration into the list

for i in range(iteration_limit):
    population = oneGeneration(population, eliteSize)
    distances = [Fitness(p).routeDistance() for p in population]
    min_dist = min(distances)
    print("Best distance for population in iteration " + str(i) +
          ": " + str(min_dist))
    progress.append(1 / rankRoutes(population)[0][1])#append the best route in each iteration into the list

#plot the graph showing the improvement of solution over generations
plt.plot(progress)
plt.ylabel('Distance')
plt.xlabel('Generation')
plt.show()

end_time = time.time()
print("Time taken: {} s".format(end_time-start_time))


# ## Saving the final solution
# 
# Once you have completed the lab, you will have to save the final solution to a CSV file for verification. Note that any cheating (identical CSV files, reporting wrong total distances, or modifying coordinates) will result in zero marks awarded for this lab.

# In[ ]:


filename = 'mysolution.csv'
distances = [Fitness(p).routeDistance() for p in population]
index = np.argmin(distances)
best_route = population[index]
with open(filename, mode='w') as f:
    writer = csv.writer(f, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(best_route)):
        writer.writerow([i, best_route[i].x, best_route[i].y])

