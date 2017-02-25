from being import Being
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from math import pi, cos, sin, acos, sqrt

"""
Map.beings_list est une liste contenant tous
les individus Being, sa longueur est donc de quelques milliers.
Chaque objet Being contient plusieurs paramètres : move_range,
satiation, type_id etc.

Pour optimiser le programme nous utiliserons beings_map:

beings_map est un numpy.array de taille : (width,height).
beings_map[i][j] est un  set d'indices des individus issu de <code>beings_list</code> avec
leurs <code>positions</code> contenu dans un carré de côté 1, d'origine coin supérieur gauche : (i,j)

free_indexes_set est l'ensemble des indices libre.

plant_count_map[i][j] = le nombre de plantes contenues dans le carré de côté 1,
d'origine coin supérieur gauche : (i,j)
"""

class Map:
    def __init__(self, beings_count, width, height, species_list):
        """
        """
        self.plant_count_map = np.zeros((width, height))
        self.beings_map = np.zeros((width, height)).astype(set)
        for i in range(width):
            for j in range(height):
                self.beings_map[i, j] = set()
        self.beings_list = []
        self.free_indexes_set = set()
        for i in range(len(beings_count)):
            for j in range(beings_count[i]):
                self.create_being(species_list[i], True)
                
    def create_being(self, species, random_position, position=None):
        if random_position:
            position = np.array((np.random.rand() * self.beings_map.shape[0],
                    np.random.rand() * self.beings_map.shape[1]))
        else:
            position = position.copy()
        being = Being(species,
                      position,
                      self.beings_map.shape)
        if (len(self.free_indexes_set) == 0):
            being.set_position_in_list(len(self.beings_list))
            self.beings_list.append(being)
        else:
            being.set_position_in_list(self.free_indexes_set.pop())
            self.beings_list[being.get_position_in_list()] = being
        self.add_being_to_map(being)
            
    def remove_being_from_map(self, being):
        self.beings_map[being.get_int_position()].remove(being.get_position_in_list())
        if being.is_plant():
            self.plant_count_map[being.get_int_position()] -= 1
    
    def add_being_to_map(self, being):
        self.beings_map[being.get_int_position()].add(being.get_position_in_list())
        if being.is_plant():
            self.plant_count_map[being.get_int_position()] += 1
        
    def move_being(self, being, roam=True, angle=0):
        if roam:
            angle = np.random.rand() * 2 * pi
            if np.random.rand() < being.direction_change_probability:
                being.current_direction = being.move_range * np.array((cos(angle), sin(angle)))
        self.remove_being_from_map(being)
        if roam:
            being.modify_position(
                                     (
                                         cos(angle) * being.move_range * being.random_movement_portion
                                         + being.current_direction[0] * (1 - being.random_movement_portion),

                                         sin(angle) * being.move_range * being.random_movement_portion
                                         + being.current_direction[1] * (1 - being.random_movement_portion)
                                     ),
                                     self.beings_map.shape
                                 )
        else:
            being.modify_position(
                                     (
                                         cos(angle) * being.move_range,
                                         sin(angle) * being.move_range,
                                     ),
                                     self.beings_map.shape
                                 )
        self.add_being_to_map(being)
        
    def iterate_ring(self, being, radius):
        """
        generateur qui retourne les cellules de carrée de coté 2*raduis + 1, centré sur la position de being
        """
        for i in range(-radius, radius + 1):
            current_cell_position = being.modify_position((i, -radius), self.beings_map.shape, change_position=False)
            yield current_cell_position, (i, -radius)
        for i in range(-radius, radius + 1):
            current_cell_position = being.modify_position((i, radius), self.beings_map.shape, change_position=False)
            yield current_cell_position, (i, radius)
        for i in range(-radius + 1, radius):
            current_cell_position = being.modify_position((-radius, i), self.beings_map.shape, change_position=False)
            yield current_cell_position, (-radius, -i)
        for i in range(-radius + 1, radius):
            current_cell_position = being.modify_position((radius, i), self.beings_map.shape, change_position=False)
            yield current_cell_position, (radius, -i)
        
    def move_towards_prey(self, being):
        if being.prey_search_range <= 1:
            return False
        for radius in range(1, being.prey_search_range + 1):
            for position, displacement in self.iterate_ring(being, radius):
                for prey_index in self.beings_map[position]:
                    if self.beings_list[prey_index].type_id in being.edible_species_id and not self.beings_list[prey_index].is_dead:
                        angle = acos(displacement[0] / sqrt(displacement[0] ** 2 + displacement[1] ** 2))
                        if displacement[1] < 0:
                            angle = -angle
                        self.move_being(being, roam=False, angle=angle)
                        return True
        return False
                        
        
        
    def delete_being(self, being):
        self.remove_being_from_map(being)
        self.free_indexes_set.add(being.get_position_in_list())
        being.is_dead = True
    
    def eat(self, being, prey):
        being.satiation += prey.satiation / 2
        self.delete_being(prey)
    
    def find_and_eat(self, being):
        for i in range(-1, 2):
            for j in range(-1, 2):
                current_cell_position = being.modify_position((i,j), self.beings_map.shape, change_position=False)
                for being_index in self.beings_map[current_cell_position]:
                    if (not self.beings_list[being_index].is_dead
                            and being.get_position_in_list() != self.beings_list[being_index].get_position_in_list()
                            and self.beings_list[being_index].type_id in being.edible_species_id
                            ):
                        self.eat(being, self.beings_list[being_index])
                        return True
        return False
    
    def reproduction(self, being):
        being.reproduction_current_cooldown = being.reproduction_cooldown * (np.random.rand() + 0.5)
        being.satiation /= 2
        if being.is_plant():
            self.create_being(being.get_species(), True)
        else:
            position = being.get_position() + np.random.rand(2) - 0.5
            self.create_being(being.get_species(), False, position=position)
    
    def iterate_being(self, being):
        if being.reproduction_current_cooldown > 0:
            being.reproduction_current_cooldown -= 1
        if being.is_plant():
            being.satiation += max(1 / self.plant_count_map[being.get_int_position()] - 0.3, 0)
        else:
            being.satiation -= 1
        if (being.satiation <= 0):
            self.delete_being(being)
            return
        if (being.satiation < being.hunger_threshold and not being.is_plant()):
            if self.find_and_eat(being):
                return
            else:
                if self.move_towards_prey(being):
                    return
        if (being.satiation > being.reproduction_threshold
                and being.reproduction_current_cooldown <= 0):
            self.reproduction(being)
        if not being.is_plant():
            self.move_being(being)

    def iteration(self):
        for being in self.beings_list:
            if not being.is_dead:
                self.iterate_being(being)