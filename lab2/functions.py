from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from functools import partial


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def selection(population, items, knapsack_max_capacity, n_selection):
    sum_fitness = sum(fitness(items, knapsack_max_capacity, individual) for individual in population)

    selected_parents = []
    for _ in range(n_selection):
        r = random.uniform(0, 1)
        total = 0
        i = 0
        while total < r:
            total += fitness(items, knapsack_max_capacity, population[i]) / sum_fitness
            i += 1
        selected_parents.append(population[i - 1][:])
    return selected_parents


def crossover(selected_parents, population_size):
    children = []
    for i in range(population_size // 2):
        random_par1 = random.choice(selected_parents)
        random_par2 = random.choice(selected_parents)

        while random_par2 == random_par1:
            random_par2 = random.choice(selected_parents)

        middle_par = len(random_par1) // 2
        children.append(random_par2[middle_par:] + random_par1[:middle_par])
        children.append(random_par1[middle_par:] + random_par2[:middle_par])
    return children


def mutation(children):
    len_children = len(children)
    for i in range(len_children):
        chosen_child = children[i]
        bit_random = random.randint(0, len(chosen_child) - 1)
        chosen_child[bit_random] = not chosen_child[bit_random]
    return children


def elitism(population, children, items, knapsack_max_capacity):
    fitness_partial = partial(fitness, items, knapsack_max_capacity)
    best_organism = max(population, key=fitness_partial)
    random_children = random.sample(children, 99)

    new_population = [best_organism] + random_children

    return new_population
