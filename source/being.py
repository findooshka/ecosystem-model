import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from math import pi, cos, sin, acos, sqrt

class Being:
    def __init__(self,
                 species,
                 position,
                 map_shape):
        """
        self (objet)
        species -> tuple(float,number,number,set(int),color,number,number,number,float,float,int)
        map_shape tuple(int,int) 
        
        Initialisation d'un objet : Being
        """
        (
            self.move_range,
            self.satiation,
            self.type_id,
            self.edible_species_id,
            self.color,
            self.reproduction_cooldown,
            self.reproduction_threshold,
            self.hunger_threshold,
            self.direction_change_probability,
            self.random_movement_portion,
            self.prey_search_range
        ) = species
        self.reproduction_current_cooldown = np.random.rand() * self.reproduction_cooldown
        self.set_position(position, map_shape)
        self.current_direction = self.move_range * (np.random.rand(2) - 1/2)
        self.is_dead = False
        self.position_in_list = None
    
    def set_position_in_list(self, position):
        """
        position -> int
        position dans la Map.being.list 
        
        Change la variable depuis la position dans Being
        """
        self.position_in_list = position
    
    def get_position_in_list(self):
        if self.position_in_list == None:
            raise RuntimeError("Position in list not initialized")
        return self.position_in_list
    
    def is_plant(self):
        return len(self.edible_species_id) == 0
    
    def get_int_position(self):
        """
        Transforme la position (i,... ; j,...) en -> int (i, j)
        """
        return tuple(self.position.astype(int))
    
    def get_position(self):
        return tuple(self.position)
    
    def format_position(self, shape, position):
        """
        Fonction pour modéliser un monde de Tore, qui change la position d'un individu
        s'il dépasse les extrémités
        """
        position = position.copy()
        if (position[0] >= shape[0]):
            position[0] -= shape[0]
        if (position[1] >= shape[1]):
            position[1] -= shape[1]
        if (position[0] < 0):
            position[0] += shape[0]
        if (position[1] < 0):
            position[1] += shape[1]
        return position
    
    def modify_position(self, addition, map_shape, change_position=True):
        """
        Lorsque change_position=True, change la position.
        Sinon on renvoie sa position + addition
        """
        if change_position:
            self.position = self.format_position(map_shape, self.position + addition)
            return
        return tuple(self.format_position(map_shape, self.position + addition).astype(int))
        
    def set_position(self, new_position, map_shape):
        self.position = self.format_position(map_shape, new_position)
        
    def get_species(self, evolution_rate=0):
        """
        23/02/17 : fonction d'évolution des paramètres non utilisé pour l'instant
        """
        if evolution_rate == 0:
            return (
                       self.move_range,
                       self.satiation,
                       self.type_id,
                       self.edible_species_id.copy(),
                       self.color,
                       self.reproduction_cooldown,
                       self.reproduction_threshold,
                       self.hunger_threshold,
                       self.direction_change_probability,
                       self.random_movement_portion,
                       self.prey_search_range,
                   )
        else:
            return (
                       self.move_range * (1 + 2 * evolution_rate * (np.random.rand()-0.5)),
                       self.satiation,
                       self.type_id,
                       self.edible_species_id.copy(),
                       self.color,
                       self.reproduction_cooldown * (1 + 2 * evolution_rate * (np.random.rand()-0.5)),
                       self.reproduction_threshold * (1 + 2 * evolution_rate * (np.random.rand()-0.5)),
                       self.hunger_threshold * (1 + 2 * evolution_rate * (np.random.rand()-0.5)),
                       self.direction_change_probability,
                       self.random_movement_portion,
                       self.prey_search_range,
                   )



