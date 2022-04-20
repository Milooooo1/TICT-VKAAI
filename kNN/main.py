import sys
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], 
                     converters={5: lambda s: 0 if s == b"-1" else float(s), 
                                 7: lambda s: 0 if s == b"-1" else float(s)}
                    )

dates = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[0])

labels = []
for label in dates:
  if label < 20000301:
    labels.append("winter")
  elif 20000301 <= label < 20000601:
    labels.append("lente")
  elif 20000601 <= label < 20000901:
    labels.append("zomer")
  elif 20000901 <= label < 20001201:
    labels.append("herfst")
  else: # from 01-12 to end of year
    labels.append("winter")

validation_data = np.genfromtxt("validation1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], 
                     converters={5: lambda s: 0 if s == b"-1" else float(s), 
                                 7: lambda s: 0 if s == b"-1" else float(s)}
                    )

validation_dates = np.genfromtxt("validation1.csv", delimiter=";", usecols=[0])

validation_labels = []
for label in validation_dates:
  if label < 20000301:
    validation_labels.append("winter")
  elif 20000301 <= label < 20000601:
    validation_labels.append("lente")
  elif 20000601 <= label < 20000901:
    validation_labels.append("zomer")
  elif 20000901 <= label < 20001201:
    validation_labels.append("herfst")
  else: # from 01-12 to end of year
    validation_labels.append("winter")

# Normalize validation and test data
normalized_data = np.array([[((i - min(day)) / (max(day) - min(day))) for i in day] for day in data])
normalized_val_data = np.array([[((i - min(day)) / (max(day) - min(day))) for i in day] for day in validation_data])

print("Data normalized.")

# Go through all validation data points
correct = 0
incorrect = 0
k = int(sys.argv[1])
verbose = False
for index, data_point in enumerate(normalized_val_data):
  
  # Get all the distances
  distances = [math.dist(reference_point, normalized_val_data[index]) for reference_point in normalized_data]
  
  # Sort the distances but keep the original index
  org_distance = copy.deepcopy(distances)
  distances.sort()
  # Get only the k shortest distances
  nearest_neighbor_distances = distances[:k]
  # Get the indexes of these distanced based on their original index before sorting
  nearest_neighbor_ids = [org_distance.index(dist) for dist in nearest_neighbor_distances]
  # Get the labels for the specific days
  nearest_neighbor_labels = [labels[id] for id in nearest_neighbor_ids]

  if verbose:
    print(f"Validation data point number: {index}")
    [print(f"Neighbor distance: {round(org_distance[id], 3)} with label: {labels[id]} vs ground truth: {validation_labels[index]}")
            for id in nearest_neighbor_ids]
    print(f"Most common neighbor: {max(nearest_neighbor_labels)}\n")
  
  # Keep track of correct estimations
  if max(nearest_neighbor_labels) == validation_labels[index]: 
    correct += 1
  else: 
    incorrect += 1

print(f"Accuracy: {(correct / len(normalized_val_data) * 100)}% with k: {k}")