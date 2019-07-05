#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithm Lab

# This notebook is meant to guide you in your first full program for the Artificial Intelligence course. Instructions and convenience classes are prepared for you, but you will need to fill in various code cells in order for the notebook to be fully-functioning. These code cells are marked with #TODO comments. Feel free to modify any other code in this notebook as well. In particular, wherever you see #SUGGESTION comments, you may want to explore alternatives (not compulsory).
# 
# The problem to be solved in this lab is the Travelling Salesman Problem. More details on this problem are provided in your lab sheet.

# ## Imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Put the imports you need here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import csv
from pprint import pprint as print # pretty printing, easier to read but takes more room
#import operator


# ## Convenience Classes
# 
# The 'City' class allows us to easily measure distance between cities. A list of cities is called a route, and will be our chromosome for this genetic algorithm.

# In[ ]:


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


# The 'Fitness' class helps to calculate both the distance and the fitness of a route (list of City instances).

# In[ ]:


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


# ## Initialization Step
# 
# Initialization starts with a large **population** of randomly generated chromosomes. We will use 3 functions. The first one generates a list of cities from a file.

# In[ ]:


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


# In[ ]:


###Testing
genCityList("TSPdata/tsp-case01.txt")


# The second function generates a random route (chromosome) from a list of City instances.

# In[ ]:


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# The third function repeatedly calls the second function to create an initial population (list of routes).

# In[ ]:


def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    #SUGGESTION - Could population be 'seeded' with known good routes?
    # In other words, would heuristic initialization help?
    return population


# The cells below are set to have Markdown type, even though they contain python code. You should change their type and run them to test the functions you've created in this section. You can always change any cell's type to Markdown to 'disable' it from running with everything else.

# cityList = genCityList('tsp-case00.txt')
# print(cityList)

# cityList = genCityList('tsp-case01.txt')
# population = initialPopulation(3, cityList)
# print(population)

# In[ ]:


###Testing
cityList = genCityList('TSPdata/tsp-case01.txt')
population = initialPopulation(10, cityList)
print(population)


# ## Selection
# 
# Parent selection is the primary form of selection, and is used to create a mating pool.

# In[ ]:


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    #return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    # lambda x : x[1] will return the value of an item in the dict
    return sorted(fitnessResults.items(), key = lambda x : x[1], reverse = True)###Possible Error


# In[ ]:


###Testing
popRanked = rankRoutes(population)


# In[ ]:


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
    ###1st approach - Rank###
    df = pd.DataFrame(np.array(popRanked), columns=['Index', "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()# calculate the cumulative sum
    df['cum_perc'] = 100 * df.cum_sum/df.Fitness.sum()# convert the cum_sum to percentage form
    selectionResults = []
    print(df)
    
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
    
    return matingPool


# In[ ]:


###Testing - Parent Selection Rank
parentSelection(population, popRanked)


# In[ ]:


def parentSelection(population, poolSize=None):
    """
    You can choose to run this cell or the previous one in order to
    'select' a crossover method. You can also add more cells.
    """
    
    if poolSize == None:
        poolSize = len(population)
        
    matingPool = []
    
    #TODO - implement this function by replacing the code between the TODO lines
    matingPool = population[0:poolSize]
    #TODO - the code above just selects the first N City instances.
    # Replace it with code which implements one of the parent selection
    # strategies mentioned in the lecture.
    
    return matingPool


# Another form of selection is survivor selection, which is used to ensure certain individuals (normally high fitness ones) survive to the next generation.

# In[ ]:


def survivorSelection(popRanked, eliteSize):
    """
    This function returns a list of length eliteSize (the selected
    City instances which will be preserved)
    """
    
    elites = []
    
    #TODO - implement this function by replacing the code between the TODO lines
    for i in range(eliteSize):
        elites.append(popRanked[i])
    #TODO - the code above just selects the first eliteSize City instances.
    # Replace it with code which selects the best individuals
    #SUGGESTION - age-based survivor selection isn't trivial to implement
    # based on this notebook, as you would need to make changes to how
    # the chromosomes are stored. Consider it a fun challenge (not
    # required, no bonus marks) for those who find this lab too easy.
    
    return elites


# The cells below are set to have Markdown type, even though they contain python code. You should change their type and run them to test the functions you've created in this section. You can always change any cell's type to Markdown to 'disable' it from running with everything else.

# population = initialPopulation(4, genCityList('tsp-case00.txt'))
# matingpool = parentSelection(population, 3)
# print('Initial population')
# print(population)
# print('Mating pool')
# print(matingpool)

# population = initialPopulation(4, genCityList('tsp-case00.txt'))
# elites = survivorSelection(population, 1)
# print('Initial population')
# print(population)
# print('Selected elites')
# print(elites)

# In[ ]:


population = initialPopulation(4, genCityList('TSPdata/tsp-case00.txt')) 
matingpool = parentSelection(population, 3)
print('Initial population')
print(population)
print('Mating pool')
print(matingpool)

population = initialPopulation(4, genCityList('TSPdata/tsp-case00.txt')) 
elites = survivorSelection(population, 1) 
print('Initial population') 
print(population)
print('Selected elites') 
print(elites)


# ## Crossover
# 
# The crossover function combines two parents in such a way that their children inherit some of each parent's characteristics. In the case of TSP, you will need to use crossover methods such as Davis' Order Crossover (other examples are listed in the lecture slides).

# In[ ]:


def crossover(parent1, parent2):
    """
    Note that this function returns TWO routes. Some crossover methods
    may only generate one child, in that case run the algorithm twice
    """
    
    #TODO - implement this function by replacing the code between the TODO lines
    child1 = parent1
    child2 = parent2
    #TODO - the code above simply returns the parents (no change). Replace
    # it with code which implements a suitable crossover method.
    
    return child1, child2


# In[ ]:


def crossover(parent1, parent2):
    """
    You can choose to run this cell or the previous one in order to
    'select' a crossover method. You can also add more cells.
    """
    
    #TODO - implement this function by replacing the code between the TODO lines
    child1 = createRoute(parent1)
    child2 = createRoute(parent2)
    #TODO - the code above simply generates new random routes.
    # Replace it with code which implements a suitable crossover method.
    
    return child1, child2


# Crossover should be run on pairs from the mating pool to produce a new generation (of the same size).

# In[ ]:


def breedPopulation(matingpool):
    children = []
    
    for i in range(1, len(matingpool), 2):
        child1, child2 = crossover(matingpool[i-1], matingpool[i])
        children.append(child1)
        children.append(child2)
    #SUGGESTION - would randomly choosing parents from matingpool make
    # a difference compared to just choosing them in order? Wouldn't be
    # too hard to test that, would it?
    
    return children


# The cells below are set to have Markdown type, even though they contain python code. You should change their type and run them to test the functions you've created in this section. You can always change any cell's type to Markdown to 'disable' it from running with everything else.

# population = initialPopulation(2, genCityList('tsp-case00.txt'))
# parent1, parent2 = population
# child1, child2 = crossover(parent1, parent2)
# print('Parents')
# print(parent1)
# print(parent2)
# print('Children')
# print(child1)
# print(child2)

# ## Mutation
# 
# Mutations are small random changes which maintain/introduce diversity. By necessity, mutations must occur at low probability and avoid changing everything in a chromosome. As with crossover, mutation in TSP must respect the constraint that every City occurs exactly once in the Route.

# In[ ]:


def mutate(route, mutationProbability):
    """
    mutationProbability is the probability that any one City instance
    will undergo mutation
    """
    mutated_route = route[:]
    for i in range(len(route)):
        if (random.random() < mutationProbability):
            #TODO - implement this function by replacing the code between
            # the TODO lines
            city1 = route[i]
            city2 = route[i-1]
            mutated_route[i] = city2
            mutated_route[i-1] = city1
            #TODO - the code above simply swaps a city with the city
            # before it. This isn't really a good idea, replace it with
            # code which implements a better mutation method

    return mutated_route


# In[ ]:


def mutate(route, mutationProbability):
    """
    You can choose to run this cell or the previous one in order to
    'select' a mutate method. You can also add more cells.
    """
    mutated_route = route[:]
    for i in range(len(route)):
        if (random.random() < mutationProbability):
            #TODO - implement this function by replacing the code between
            # the TODO lines
            city1 = route[i]
            city2 = route[0]
            mutated_route[i] = city2
            mutated_route[0] = city1
            #TODO - the code above simply swaps the city with the first
            # city. This isn't really a good idea, replace it with
            # code which implements a better mutation method

    return mutated_route


# The mutate function needs to be run over the entire population, obviously.

# In[ ]:


def mutation(population, mutationProbability):
    mutatedPopulation = []
    for i in range(0, len(population)):
        mutatedIndividual = mutate(population[i], mutationProbability)
        mutatedPopulation.append(mutatedIndividual)
    return mutatedPopulation


# The cells below are set to have Markdown type, even though they contain python code. You should change their type and run them to test the functions you've created in this section. You can always change any cell's type to Markdown to 'disable' it from running with everything else.

# route = genCityList('tsp-case00.txt')
# mutated = mutate(route, 1)  # Give a pretty high chance for mutation
# print('Original route')
# print(route)
# print('Mutated route')
# print(mutated)

# ## Running One Generation
# 
# Now that we have (almost) all our component functions in place, let's call them altogether.

# In[ ]:


def oneGeneration(population, eliteSize, mutationProbability):
    
    # First we preserve the elites
    elites = survivorSelection(population, eliteSize)
    
    # Then we calculate what our mating pool size should be and generate
    # the mating pool
    poolSize = len(population) - eliteSize
    matingpool = parentSelection(population, poolSize)
    #SUGGESTION - What if the elites were removed from the mating pool?
    # Would that help or hurt the genetic algorithm? How would that affect
    # diversity? How would that affect performance/convergence?
    
    # Then we perform crossover on the mating pool
    children = breedPopulation(matingpool)
    
    # We combine the elites and children into one population
    new_population = elites + children
    
    # We mutate the population
    mutated_population = mutation(new_population, mutationProbability)
    #SUGGESTION - If we do mutation before selection and breeding, does
    # it make any difference?
    
    return mutated_population


# The cells below are set to have Markdown type, even though they contain python code. You should change their type and run them to test the functions you've created in this section. You can always change any cell's type to Markdown to 'disable' it from running with everything else.

# population = initialPopulation(5, genCityList('tsp-case00.txt'))
# eliteSize = 1
# mutationProbability = 0.01
# new_population = oneGeneration(population, eliteSize, mutationProbability)
# print('Initial population')
# print(population)
# print('New population')
# print(new_population)

# ## Running Genetic Algorithm
# 
# The entire genetic algorithm needs to initialize a Route of City instances, then iteratively generate new generations. Take note that, unlike all the cells above, the cell below is NOT a function. Various parameters are set right at the top (you should set them to something reasonable).

# In[ ]:


start_time = time.time()
filename = 'tsp-case03.txt'
popSize = 20
eliteSize = 5
mutationProbability = 0.01
iteration_limit = 100

cityList = genCityList(filename)

population = initialPopulation(popSize, cityList)
distances = [Fitness(p).routeDistance() for p in population]
min_dist = min(distances)
print("Best distance for initial population: " + str(min_dist))

for i in range(iteration_limit):
    population = oneGeneration(population, eliteSize, mutationProbability)
    distances = [Fitness(p).routeDistance() for p in population]
    min_dist = min(distances)
    print("Best distance for population in iteration " + str(i) +
          ": " + str(min_dist))
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

