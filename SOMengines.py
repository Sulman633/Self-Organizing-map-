import glob
import math
import numpy as np
import random
from graphics import *
import MapNode

file_list = glob.glob("*.out")

np.set_printoptions(suppress=True)





class self_organizing_maps:
    def __init__(self, training_data, vector_size, min_value, max_value, epochs,  learning_rate_constant):

        self.training_data = training_data

        self.input_vector = 0.0

        self.vector_size = vector_size

        self.lattice = np.empty((15, 15), dtype=object)
        #holds all euclidian distances at the current iteration in aa array
        self.eucli_distances = np.empty((15, 15))
        #holds all actual pythagoran distances
        self.distances_from_BMU = np.empty((15, 15))
        #holds the weight vectors classifications per epoch, updates constantly
        self.classification_lattice = np.full((15, 15), None)

        self.epochs = epochs
        #this is a dynamic learning which decays each epoch
        self.learning_rate = 0

        self.learning_rate_constant = learning_rate_constant

        self.map_radius = len(self.lattice) / 2

        self.time_constant = self.epochs / self.map_radius

        self.neighbourhood_radius = 0.0
        #Holds the current BMU coordinates 
        self.BMU = None

        self.influence_of_distance = 0.0

        self.accuracy = 0.0
        #fills the lattice with random weights between the min and max value of the input data coming in.
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                self.lattice[i][j] = MapNode.map_node(vector_size, min_value, max_value).get_weights()

    #The euclidian distance formula squared used as a means to meaure the distance between input vector and random weight vector.
    def get_distance(self, input_vector, weight_vector):
        distance = 0.0
        #deletes the identifier in the starting of the vector
        self.input_vector = np.delete(input_vector, 0)
        distance_vector = (self.input_vector - weight_vector) ** 2
        distance = np.sum(distance_vector)
        return np.sqrt(distance)

    #Finds the BMU by searching through the array that holds all the euclidian distances
    def find_BMU(self, input_vector):
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                self.eucli_distances[i][j] = self.get_distance(input_vector, self.lattice[i][j])
        #finds the BMU coordinates in the lattice
        self.BMU = np.argwhere(self.eucli_distances == np.amin(self.eucli_distances))

    #shrinks the radius as t gets larger and larger.
    def calculate_neighbourhood_radius(self, t):
        self.neighbourhood_radius = self.map_radius * np.exp(-t / self.time_constant)
        return self.neighbourhood_radius
        
    #calculates the distances for each node from the BMU
    def dist_from_BMU(self):
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                #BMUi - i, BMUj - j
                dist_from_BMU = np.sqrt(((self.BMU[0][0] - i) ** 2) + ((self.BMU[0][1] - j) ** 2))
                self.distances_from_BMU[i][j] = dist_from_BMU

    #represent the amount of influence a node's distance from the BMU has on its learning using gaussian function or mexican hat function
    def influenece_functions(self, t, i, j, func = False):
        if func == True:
            temp = (1 - (self.distances_from_BMU[i][j]** 2) / self.calculate_neighbourhood_radius(t) ** 2) 
            numerator = -self.distances_from_BMU[i][j] ** 2
            denom = 2 * self.calculate_neighbourhood_radius(t) ** 2
            self.influence_of_distance = temp * np.exp(numerator / denom)
        else: 
            numerator = -self.distances_from_BMU[i][j] ** 2
            denom = 2 * self.calculate_neighbourhood_radius(t) ** 2
            self.influence_of_distance = np.exp(numerator / denom)

    #updates the weights based on the influenece_functions or mexican hat. Does this by going through all weight vectors in the lattice array.
    def update_weights(self, t):
        self.learning_rate = self.learning_rate_constant * np.exp(-t / self.epochs)
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                self.influenece_functions(t, i, j, False) #*********Set to True if you want to use mexican hat function*******
                vector = self.lattice[i][j]
                updated_weight = self.lattice[i][j] + ((self.influence_of_distance * self.learning_rate) * (self.input_vector - self.lattice[i][j]))
                self.lattice[i][j] = updated_weight
        
    # marks nodes with a 0 or a 1 if they are inside the neighbourhood radius depending on the input_vector's identifier.
    def mark_nodes(self, input_vector):
        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                if self.distances_from_BMU[i][j] < self.neighbourhood_radius:
                    self.classification_lattice[i][j] = input_vector[0]

    # The main training loop for the SOM, every epoch one input vector is fed in and BMU is found, then mark all the nodes that are in
    # the current neighbourhood radius of the BMU as a 0 or 1 depending on the input_vector. This example does not use holdout, so all 53 samples 
    # are used in testing the accuracy of the SOM each epoch
    def train_SOM(self):
        for i in range(self.epochs):
            print("----------------EPOCH--------------------: ", i)
            random_input_vector = random.sample(range(len(self.training_data)), len(self.training_data))
            #selects a random input_vector.
            self.find_BMU(self.training_data[random_input_vector[0]])
            self.calculate_neighbourhood_radius(i)
            self.dist_from_BMU()
            self.mark_nodes(self.training_data[random_input_vector[0]])
            self.update_weights(i)
            self.test_accuracy()
            #break if 95% accuracy is reached.
            if self.accuracy >= 0.80:
                break
        #print("Classification vector: \n", self.classification_lattice)
        print("Lambda: ", self.time_constant)

    #Same as train_SOM but holds 15% of the data for testing the SOM on each epoch like before, so only 15% is trained on
    #instead of 53 samples like before.
    def holdout_train_SOM(self):
        #use 85% of the data only .
        random_input_vector = random.sample(range(len(self.training_data)), len(self.training_data))
        holdout_amount = int(0.15 * len(random_input_vector))
        #holds these values for testing.
        held_input = random_input_vector[0:holdout_amount]
        #deletes the split_input vector from the original list.
        del random_input_vector[0:holdout_amount]

        for i in range(self.epochs):
            print("----------------EPOCH--------------------: ", i)
            random.shuffle(random_input_vector)
            self.find_BMU(self.training_data[random_input_vector[0]])
            self.calculate_neighbourhood_radius(i)
            self.dist_from_BMU()
            self.mark_nodes(self.training_data[random_input_vector[0]])
            self.update_weights(i)
            self.holdout_test_accuracy(held_input)
            if self.accuracy >= 0.80:
                break
        print("Lambda: ", self.time_constant)
    # Seperate method is the same as the other find_BMU method but local to test_accuracy methods to avoid 
    # conflicts.
    def test_find_BMU(self, input_vector):
        eucli_distances = np.empty((15, 15))

        for i in range(len(self.lattice)):
            for j in range(len(self.lattice)):
                eucli_distances[i][j] = self.get_distance(input_vector, self.lattice[i][j])
        #finds the BMU coordinates in the lattice
        BMU = np.argwhere(eucli_distances == np.amin(eucli_distances))
        return BMU

    # checks the accuracy by concatenating the BMU coordinates and identifier and then checking if its stored already or not
    # after that check to see if the input vector identifier is the same as the current BMU's identifier and if so then increment
    # correct which is the counter to see how many are right.
    def test_accuracy(self):
        stored_BMUs = [''] * len(self.training_data)
        correct = 0
        self.accuracy = 0.0
        stored = False
        good = 0
        bad = 0

        for i in range(len(self.training_data)):
            input_vector = self.training_data[i][0]
            identifier = np.array_str(input_vector)
            BMU = self.test_find_BMU(self.training_data[i])
            BMU1 = str(BMU[0][0])
            BMU2 = str(BMU[0][1])
            # in the form of " 1 3 0.0"
            BMU_identifier = BMU1 + " " + BMU2 + " " + identifier

            #Break for loop if the same BMU coordinates are stored already
            for j in range(len(self.training_data)):
                if BMU_identifier == stored_BMUs[j]:
                    stored = True
            if stored:
                continue
            else:
                stored_BMUs[i] = BMU_identifier
                stored = False
                #check to see if both identifiers are the same.
                if self.classification_lattice[BMU[0][0]][BMU[0][1]] == input_vector:
                    if input_vector == 0:
                        good += 1
                    else:
                        bad += 1
                    correct += 1
        
        print("Good motors: ", good)
        print("Bad motors: ", bad)
        self.accuracy = correct / len(self.training_data)
        print("Accuracy: ", self.accuracy)
    
    #same as test_accuracy method
    def holdout_test_accuracy(self, input):
        stored_BMUs = [''] * len(input)
        correct = 0
        self.accuracy = 0.0
        stored = False
        good = 0
        bad = 0

        for i in range(len(input)):
            input_vector = self.training_data[input[i]][0]
            identifier = np.array_str(input_vector)
            BMU = self.test_find_BMU(self.training_data[input[i]])
            BMU1 = str(BMU[0][0])
            BMU2 = str(BMU[0][1])
            BMU_identifier = BMU1 + " " + BMU2 + " " + identifier

            #Break for loop if the same BMU coordinates are stored already
            for j in range(len(input)):
                if BMU_identifier == stored_BMUs[j]:
                    stored = True
            if stored:
                continue
            else:
                stored_BMUs[i] = BMU_identifier
                stored = False
                if self.classification_lattice[BMU[0][0]][BMU[0][1]] == input_vector:
                    if input_vector == 0:
                        good += 1
                    else:
                        bad += 1
                    correct += 1
        print("Good motors: ", good)
        print("Bad motors: ", bad)
        self.accuracy = correct / len(input)
        print("Accuracy: ", self.accuracy)

    # makes a simple GUI using the classificattion arrayand tells how many motors are good and bad
    def create_GUI(self):
        win = GraphWin('SOM', 800, 800)
        win.setCoords(0, 15, 15, 0)
        win.setBackground("white")

        for i in range(len(self.classification_lattice)):
            for j in range(len(self.classification_lattice)):
                square = Rectangle(Point(j, i), Point(j + 1, i + 1))

                if self.classification_lattice[i][j] == 0.0:
                    square.setFill("green")
                    square.draw(win)
                elif self.classification_lattice[i][j] == 1.0:
                    square.setFill("red")
                    square.draw(win)
                else:
                    square.setFill("white")
                    square.draw(win)
        win.getMouse()

#loads the specified data file.
def load_file(file):
    data = np.genfromtxt(file_list[file], delimiter=" ", skip_header=1)
    return data

#takes the first line of the file and saves the information about how long the vector is and how many there are.
def create_header(file):
    with open(file_list[file], 'r') as f:
        first_line = f.readline()
        first_line = list(map(int, first_line.split()))
    return first_line

#The main method where I load the files first and find the min and max values of the file currently selected. 
#Then create a SOM's object and run the either the normal training or the holdout training
#By changing load file from 0-3 you can cycle through the data files where 0 is L30fft16, 1 is L30fft25, 2 isL30fft32 and so forth.
# ******To use the mexican hat you must set the parameter for self.influenece_functions to TRUE, this method is located in the update_weights method****
#********To change epochs or learning rate, change the last wo parameters of SOM***********
def main():
    data = load_file(0)
    max_value = np.amax(data)
    min_value = np.amin(data)
    header = create_header(0)
    SOM = self_organizing_maps(data, header[1], min_value, max_value, 300, 2.3)
    #SOM.train_SOM()
    SOM.holdout_train_SOM()
    SOM.create_GUI()

if __name__ == "__main__":
    main()