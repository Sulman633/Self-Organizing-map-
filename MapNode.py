import glob
import math
import numpy as np

class map_node:

    def __init__(self, vector_size, min_value, max_value):
        
        self.weight_vector = np.random.uniform(min_value, max_value, size=(1, vector_size))

    def get_weights(self): 
        return self.weight_vector