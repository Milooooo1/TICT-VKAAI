import numpy as np
import math
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
normalized_data = preprocessing.normalize(np.array(data))
normalized_val_data = preprocessing.normalize(np.array(validation_data))

print("Data normalized.")

# Go through all validation data points
correct = 0
incorrect = 0
for index, data_point in enumerate(normalized_val_data):
  data_point = normalized_val_data[index]

  # Get all the distances between the new data point and the test data
  distances = np.linalg.norm(normalized_data - data_point, axis=1)

  # Find the nearest neighbours and get the labels
  k = 5
  nearest_neighbor_ids = distances.argsort()[:k]
  nearest_neighbor_labels = [labels[id] for id in nearest_neighbor_ids]
  if max(nearest_neighbor_labels) == validation_labels[index]: 
    correct += 1
  else: 
    incorrect += 1

print(f"Accuracy: {(correct / len(normalized_val_data) * 100)}%")