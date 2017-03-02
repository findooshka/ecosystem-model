import signal
import sys, os
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import source.map
import source.measurement
import source.random_forest


def sigint_handler(signum, frame):
    global tests_filename
    global tests_df
    if tests_filename is None:
        tests_filename = 'tests' + \
                         time.strftime('-%m-%d-%H', time.localtime()) + \
                         '.csv'
        tests_filename = os.path.join("data", tests_filename)
        if tests_df.shape[0] != 0:
            tests_df.to_csv(tests_filename, index=False)
            print('written {} tests to {}'.format(tests_df.shape[0], tests_filename))
        else:
            print('zero tests done')

    raise KeyboardInterrupt()


def initialize_map():
    species = [(0.05, 130, 0, {2}, (0, 0, 1, 1), 200, 101, 100, 0.01, 0.15, 0),
           (0.15, 130, 1, {0}, (1, 0, 0, 1), 400, 151, 150, 0.01, 0.15, 3),
           (0, 10, 2, set(), (0, 0.8, 0.2, 1), 50, 40, 1000, 0.01, 0.15, 0),
          ]

    return source.map.Map([100, 5, 1500], 50, 50, species)


def make_one_test(map_size,
                  max_iterations,
                  test_count,
                  inital_beings_number_bounds,
                  simulations_per_test=1,
                  change_satiation_threshold=True,
                  change_reproduction_threshold=True,
                  change_reproduction_cooldown=True,
                  change_inital_satiation=True,
                  change_hunt_range=False):
    global species
    return_data = []
    iterations_survived = None
    species_count = 0
    species_count_list = []
    current_species = np.array(species.copy())
    if change_satiation_threshold:
        current_species[:, 7] *= (0.5 + 2 * np.random.rand(current_species.shape[0]))
    if change_reproduction_threshold:
        current_species[:, 6] *= (0.5 + 2 * np.random.rand(current_species.shape[0]))
    if change_reproduction_cooldown:
        current_species[:, 5] *= (0.5 + 2 * np.random.rand(current_species.shape[0]))
    if change_inital_satiation:
        current_species[:, 1] *= (0.5 + 2 * np.random.rand(current_species.shape[0]))
    if change_hunt_range:
        current_species[:, 10] += np.random.randint(-2, 4, current_species.shape[0])
    inital_count = []
    for bounds in inital_beings_number_bounds:
        inital_count.append(np.random.randint(bounds[0], bounds[1] + 1))
        if inital_count[-1] != 0:
            species_count += 1
    species_count_list += [inital_count]
    the_map = source.map.Map(inital_count, map_size, map_size, current_species)
    return_data = np.concatenate(current_species[:, [1, 5, 6, 7, 10]]).tolist()
    for iteration in range(max_iterations):
        if not source.measurement.survival_simulation(the_map, species_count):
            iterations_survived = iteration
            break
    else:
        iterations_survived = max_iterations
    print(species_count_list)
    return species_count_list + return_data + [iterations_survived]


def main():
    global tests_df
    test_count = 0
    while True:
        test = make_one_test(50,
                            max_iterations=6000,
                            test_count=10000,
                            inital_beings_number_bounds=[(50, 400), (2, 10), (1000, 6000)],
                            simulations_per_test=1,
                            change_satiation_threshold=True,
                            change_reproduction_threshold=True,
                            change_reproduction_cooldown=True,
                            change_inital_satiation=True,
                            change_hunt_range=True)
        #test = source.measurement.survival_tests(os.path.join("data", "tests.csv"), # doesn't mean anything, see sigint_handler
        #                                       species,
        #                                       50,
        #                                       max_iterations=6000,
        #                                       test_count=1,
        #                                       inital_beings_number_bounds=[(50, 400), (2, 10), (1000, 6000)],
        #                                       simulations_per_test=1,
        #                                       change_satiation_threshold=True,
        #                                       change_reproduction_threshold=True,
        #                                       change_reproduction_cooldown=True,
        #                                       change_inital_satiation=True,
        #                                       change_hunt_range=True,
        #                                       stop_hour=9)
        tests_df = tests_df.append(test, ignore_index=True)
        test_count += 1
        print('done {} tests'.format(test_count))


tests_filename = None


species = [(0.05, 130, 0, {2}, (0, 0, 1, 1), 200, 101, 100, 0.01, 0.15, 0),
           (0.15, 130, 1, {0}, (1, 0, 0, 1), 400, 151, 150, 0.01, 0.15, 3),
           (0, 10, 2, set(), (0, 0.8, 0.2, 1), 50, 40, 1000, 0.01, 0.15, 0),
          ]


columns = ['species 0 inital count', 'species 1 inital count',
       'species 2 inital count', 'species 0 inital satiation',
       'species 0 satiation threshold', 'species 0 reproduction cooldown',
       'species 0 reproduction threshold', 'species 0 hunt range',
       'species 1 inital satiation', 'species 1 satiation threshold',
       'species 1 reproduction cooldown',
       'species 1 reproduction threshold', 'species 1 hunt range',
       'species 2 inital satiation', 'species 2 satiation threshold',
       'species 2 reproduction cooldown',
       'species 2 reproduction threshold', 'species 2 hunt range', 'result']


tests_df = pd.DataFrame(columns=columns)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    main()

#signal.signal(signal.SIGINT, sigint_handler)
#i = 0
#while True: 
#    i += 1