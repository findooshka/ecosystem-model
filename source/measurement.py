import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from math import pi, cos, sin, acos, sqrt
import source.map
import source.basic_classes

def get_population(beings_list):
    """
    list[being] -> set(int)
    Compte le nombre total d'individu pour chaque espèce dans un ensemble.
    """
    result = {}
    for being in beings_list:
        if being.is_dead:
            continue
        if not being.type_id in result.keys():
            result[being.type_id] = 1
        else:
            result[being.type_id] += 1
    return result


def draw_population(global_map,
                    iteration_number,
                    iterations_per_step,
                    species_to_draw=None,
                    stop_on_extinction=False):
    """
    faire une graphe de dynamique de population.
    """
    plots = {}
    inital_population = get_population(global_map.beings_list)
    population = get_population(global_map.beings_list)
    for type in inital_population:
        plots[type] = []
    current_iteration = 0
    extinct = False
    while current_iteration < iteration_number and not (stop_on_extinction and extinct):
        for type in inital_population:
            if not type in population:
                plots[type].append(0)
                extinct = True
            else:
                plots[type].append(population[type])
        for i in range(iterations_per_step):
            global_map.iteration()
        population = get_population(global_map.beings_list)
        current_iteration += 1
    plt.close('all')
    fig = plt.figure(figsize=(12, 6))
    for type in inital_population:
        if not species_to_draw == None and not type in species_to_draw:
            continue
        subplot = plt.subplot((len(inital_population.keys()) // 2 + 1) * 100 + 20 + 1 + type)
        subplot.set_xlim((0, iteration_number))
        subplot.plot(plots[type])
    plt.show()
    fig = plt.figure(figsize=(10, 6))
    for type in inital_population:
        if not species_to_draw == None and not type in species_to_draw:
            continue
        subplot = plt.subplot(111)
        subplot.set_xlim((0, iteration_number))
        subplot.semilogy(plots[type])
    plt.show()

def scatter_populations(species1,
                        species2,
                        global_map,
                        iteration_number,
                        iterations_per_point,
                        color_start=(0.5, 0, 0),
                        color_end=(0, 0.6, 0)):
    color_start = np.array(color_start)
    color_end = np.array(color_end)
    species1_population = []
    species2_population = []
    colors = []
    for current_iteration_number in range(iterations_per_point * iteration_number):
        global_map.iteration()
        if current_iteration_number % iterations_per_point == 0:
            population = get_population(global_map.beings_list)
            if not species1 in population or not species2 in population:
                break
            species1_population.append(population[species1])
            species2_population.append(population[species2])
            colors.append(color_start
                          + (color_end - color_start) * (current_iteration_number) / (iterations_per_point * iteration_number))
    plt.close('all')
    plt.plot(species1_population, species2_population, linewidth=1, alpha=0.5)
    plt.scatter(species1_population, species2_population, c=colors, s=10)
    plt.show()
    return species1_population, species2_population

def survival_simulation(the_map, species_count):
    """
    Map*int -> bool
    Faire une iteration et vérifie si l'on a ou non le bon nombre d'espèce au départ.
    True si c'est le cas, False sinon    
    """
    the_map.iteration()
    if len(get_population(the_map.beings_list)) < species_count:
        return False
    return True


def survival_tests(file_name,
                   species,
                   map_size,
                   max_iterations,
                   test_count,
                   inital_beings_number_bounds,
                   simulations_per_test=1,
                   change_satiation_threshold=True,
                   change_reproduction_threshold=True,
                   change_reproduction_cooldown=True,
                   change_inital_satiation=True,
                   change_hunt_range=False,
                   stop_hour=None):
    """
    Fonctions pour trouver de paramètres qui entraîne la survie de toutes les espèces
    """
    tested_species = np.concatenate(np.array(species)[:, [1, 5, 6, 7, 10]])
    tested_species = np.tile(tested_species, (test_count * simulations_per_test, 1))
    results = []
    species_count_list = []
    species_count = 0
    for test_number in range(test_count):
        #if stop_hour != None and time.localtime().tm_hour == stop_hour:
        #    tested_species = tested_species[:len(results)]
        #    break
        current_species = np.array(species.copy())
        if change_satiation_threshold:
            current_species[:, 7] *= (0.5 + 2 * np.random.rand(current_species.shape[0]))
        if change_reproduction_threshold:
            current_species[:, 6] *= (0.5 + 2 * np.random.rand(current_species.shape[0]))
        if change_satiation_threshold:
            current_species[:, 5] *= (0.5 + 2 * np.random.rand(current_species.shape[0]))
        if change_inital_satiation:
            current_species[:, 1] *= (0.5 + 2 * np.random.rand(current_species.shape[0]))
        if change_hunt_range:
            current_species[:, 10] += np.random.randint(-2, 4, current_species.shape[0])
        inital_count = []
        for bounds in inital_beings_number_bounds:
            inital_count.append(np.random.randint(bounds[0], bounds[1] + 1))
            if inital_count[-1] != 0 and test_number == 0:
                species_count += 1
        for simulation in range(simulations_per_test):
            species_count_list += [inital_count]
            the_map = source.map.Map(inital_count, map_size, map_size, current_species)
            tested_species[len(results)] = np.concatenate(current_species[:, [1, 5, 6, 7, 10]])
            for iteration in range(max_iterations):
                if not survival_simulation(the_map, species_count):
                    results.append([iteration])
                    break
            else:
                results.append([max_iterations])
    if len(results) == 0:
        return None
    try:
        tested_species = np.hstack((species_count_list, tested_species, results))
        columns = []
        for current_species in range(len(species)):
            columns += ["species " + str(current_species) + " inital count"]
        for current_species in range(len(species)):
            columns += [
                "species " + str(current_species) + " inital satiation",
                "species " + str(current_species) + " satiation threshold",
                "species " + str(current_species) + " reproduction cooldown",
                "species " + str(current_species) + " reproduction threshold",
                "species " + str(current_species) + " hunt range",            
            ]
        columns.append("result")
        return pd.DataFrame(data=tested_species, columns=columns)
    except Exception as err:
        print('failed testing with error:', err)
        return None      

def test_results_to_normalized_labeled_set(data_frame, threshold):
    result = source.basic_classes.LabeledSet(len(data_frame[data_frame.columns]) - 1)
    input_array = np.array(data_frame[data_frame.columns[:-1]])
    for column in range(input_array.shape[1]):
        input_array[:, column] = (
                                     (input_array[:, column] - input_array[:, column].min())
                                     / (input_array[:, column].max() - input_array[:, column].min())
                                 )
    labels = np.array(data_frame["result"])
    for i in range(input_array.shape[0]):
        result.add_example(input_array[i],
                           1 if labels[i] >= threshold else -1)
    return result

