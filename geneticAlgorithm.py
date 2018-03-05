# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from random import sample

class GA():
    def __init__(self, population, distances, n_genes, n_chromosomes):
        self.population = population
        self.distances = distances
        self.n_genes = n_genes
        self.n_chromosomes = n_chromosomes
   
    def fitness(self, chromosome):
        fitness = 0
        for i in range(self.n_genes-1):
            fitness += self.distances[chromosome[i]][chromosome[i+1]]
        return fitness
    
    def crossover1(self, parent1, parent2):
        cross_point = np.random.randint(1,self.n_genes-1)
        child1 = np.array(parent1)
        child2 = np.array(parent2)
        print("Cross at: " + str(cross_point))
        j1 = cross_point
        j2 = cross_point
        for i in range(cross_point, self.n_genes):
            while parent2[j1] in child1[:cross_point]:
                j1 = (j1+1)%self.n_genes
            child1[i] = parent2[j1]
            j1 = (j1+1)%self.n_genes
            
            while parent1[j2] in child2[:cross_point]:
                j2 = (j2+1)%self.n_genes
            child2[i] = parent1[j2]
            j2 = (j2+1)%self.n_genes
        
        return child1, child2

    def crossover2(self, parent1, parent2):
        cross_point1, cross_point2 = sample(range(n_genes), 2)
        if (cross_point1 < cross_point2):
            tmp = cross_point1
            cross_point1 = cross_point2
            cross_point2 = tmp
        child1 = np.array(parent1)
        child2 = np.array(parent2)
        j1 = cross_point1
        j2 = cross_point2
        for i in range(cross_point1, cross_point2):
            while parent2[j1] not in parent1[cross_point1:cross_point2]:
                j1 = (j1+1)%self.n_genes
            child1[i] = parent2[j1]
            j1 = (j1+1)%self.n_genes
            
            while parent1[j2] not in parent2[cross_point1:cross_point2]:
                j2 = (j2+1)%self.n_genes
            child2[i] = parent1[j2]
            j2 = (j2+1)%self.n_genes
        
        return np.array(child1), np.array(child2)
    
    def mutation(self, chromosome):
        mutated = np.array(chromosome)
        gene1 = np.random.randint(0,self.n_genes-1)
        gene2 = np.random.randint(gene1+1,self.n_genes)
        mutated[gene1] = chromosome[gene2]
        mutated[gene2] = chromosome[gene1]
        return mutated
    
    def deterministicSelection(self, selection_factor):
        """
        Keeps the best k=selection_factor chromosomes based on the fitness function.
        """
        self.population = [[x, self.fitness(x)] for x in self.population]
        self.population.sort(key=lambda x: x[1])
        self.population = [x[0] for x in self.population[:selection_factor]]
        
    def probabilisticSelection(self, selection_factor, RWS=False, p=[1, 0]):
        """
        Split [0, 1] in bins of size proportional to the fitness of the chromosomes and use a uniform random generator to draw a random number r in this range.
        The default method is Stochastic Universal Sampling, select the chromosomes corresponding to bins containing (r + k/selection_factor) % 1 for k = 0, ..., selection_factor - 1.
        If RWS (Roulette Wheel Selection) is True, draw a random number r for n=selection_factor times, if r lies in bin i, select the corresponding chromosome.
        The advantage of SUS over RWS is the lower variance on the set of selected samples.
        The p values are used for selection pressure using a linear fitness adjustment.
        """       
        
        total_fitness = sum([self.fitness(x) for x in self.population])
        #self.population = [[x, self.fitness(x)] for x in self.population]
        self.population = [[x, (p[0]*(total_fitness - self.fitness(x))+p[1])/(p[0]*(total_fitness*(len(self.population)-1))+p[1])] for x in self.population]
        #self.population = [[x[0], (p[0]*x[1] + p[1])/(p[0]*total_fitness + p[1])] for x in self.population]
        self.population.sort(key=lambda x: x[1])
        #print(self.population)
        #print()
        self.population = [[x[0], y] for x, y in zip(self.population, np.cumsum([x[1] for x in self.population]))]
        
        new_population = []
        
        if RWS:
            for r in np.random.rand(selection_factor):
                selected_chromosome = [x for x in enumerate(self.population) if x[1][1] >= r][0]
                new_population.append(selected_chromosome[1][0])
        else:
            r = np.random.rand()
            for i in range(selection_factor):
                selected_chromosome = [x for x in enumerate(self.population) if x[1][1] >= (r + i/selection_factor)%1][0]
                new_population.append(selected_chromosome[1][0])
                
        self.population = new_population
        
    def tournamentSelection(self, selection_factor, k=2, p=0):
        """
        Select k random chromosomes, keep the best one (highest ﬁtness function) and iterate for n=selection_factor times.
        If p != 0 and k == 2 then stochastic tournament selection is applied with probability p of getting the best chromosome.
        """
        new_population = []
        
        if p != 0 and k == 2:
            for i in range(selection_factor):
                s = sample(self.population, k)
                if np.random.rand() < p:
                    new_population.append(min([[x, self.fitness(x)] for x in s], key=lambda x: x[1])[0])
                else:
                    new_population.append(max([[x, self.fitness(x)] for x in s], key=lambda x: x[1])[0])
        else:
            for i in range(selection_factor):
                s = sample(self.population, k)
                new_population.append(min([[x, self.fitness(x)] for x in s], key=lambda x: x[1])[0])
        
        self.population = new_population
        
    def evolution(self, selection_factor, mut_prob):
        """
        Apply probabilistic selection to parents and create offsprings with crossover and mutation.
        mut_prob is the mutation probability.
        """
        self.tournamentSelection(selection_factor, p=0.7)
        offspring = []

        for i in range(selection_factor):
            p1, p2 = sample(self.population, 2)
            c1, c2 = self.crossover2(p1, p2)
            if not any([np.array_equal(c1, x) for x in self.population]):
                offspring.append(c1)
            if not any([np.array_equal(c2, x) for x in self.population]):
                offspring.append(c2)

        for x in self.population:
            if np.random.rand() <= mut_prob:
                c = self.mutation(x)
                if not any([np.array_equal(c, x) for x in self.population]):
                    offspring.append(c)

        self.population.extend(offspring)
        self.population.sort(key=lambda x: self.fitness(x))
        
        return self.population[0], self.fitness(self.population[0])

np.random.seed(2020)

n_genes = 16
#citiesx = np.random.permutation(100)
#citiesy = np.random.permutation(100)
#cities = [[citiesx[i], citiesy[i]] for i in range(n_genes)]
cities = [[0, 0], [0, 20], [0, 40], [0, 60], [0, 80],
          [80, 0], [80, 20], [80, 40], [80, 60], [80, 80],
          [20, 0], [40, 0], [60, 0],
          [20, 80], [40, 80], [60, 80]]
n_chromosomes = 300
distances = pairwise_distances(cities, metric='euclidean')
population = [np.random.permutation(n_genes) for x in range(n_chromosomes)]

plt.scatter([x[0] for x in cities], [x[1] for x in cities])
plt.show()

ga = GA(population, distances, n_genes, n_chromosomes)
best = [0, 1500]
for i in range(200):
    tmp = ga.evolution(int(n_chromosomes/2), 0.05)
    if tmp[1] < best[1]:
        best = tmp
    print("Iteration #" + str(i) + ":"  + str(ga.fitness(ga.population[0])))

citiesplot = [cities[i] for i in best[0]]
print("Best result: " + str(best[1]))
plt.plot([x[0] for x in citiesplot], [x[1] for x in citiesplot], marker='o')
plt.show()
