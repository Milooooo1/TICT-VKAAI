import numpy as np
import math
import copy
import argparse
import collections

def importCSVFile(filename):
  data = np.genfromtxt(filename, delimiter=";", usecols=[2,5,6,7], 
                      converters={5: lambda s: 0 if s == b"-1" else float(s), 
                                  7: lambda s: 0 if s == b"-1" else float(s)}
                      )

  dates = np.genfromtxt(filename, delimiter=";", usecols=[0])

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

  return labels, data

def loadNormalizedData():
  test_labels, test_data = importCSVFile("dataset1.csv")
  validation_labels, validation_data = importCSVFile("validation1.csv")

  norm_max = max([np.amax(test_data), np.amax(validation_data)])
  norm_min = min([np.amin(test_data), np.amin(validation_data)])

  # Normalize validation and test data
  normalized_test_data = np.array([[((i - norm_min) / (norm_max - norm_min)) for i in day] for day in test_data])
  normalized_val_data  = np.array([[((i - norm_min) / (norm_max - norm_min)) for i in day] for day in validation_data])

  return normalized_val_data, validation_labels, normalized_test_data, test_labels

def estimateDataPoint(k, ground_truth, data_point, normalized_data, data_labels, verbose):
  # Get all the distances
  # distances = [math.dist(reference_point, data_point) for reference_point in normalized_data]
  distances = [np.linalg.norm(reference_point - data_point) for reference_point in normalized_data]

  # Sort the distances but keep the original index
  org_distance = copy.deepcopy(distances)
  distances.sort()
  # Get only the k shortest distances
  nearest_neighbor_distances = distances[:k]
  # Get the indexes of these distances based on their original index before sorting
  nearest_neighbor_ids = [org_distance.index(dist) for dist in nearest_neighbor_distances]
  # Get the labels for the specific days
  nearest_neighbor_labels = [data_labels[id] for id in nearest_neighbor_ids]

  # Check for ties
  occurancies_dict = collections.Counter(nearest_neighbor_labels)
  # The length of the occurancy dict has to be more than 1 or there are no ties
  if (len(occurancies_dict.most_common()) > 1):
    # Only check first and second element, 1 tie is already enough to try again
    if(occurancies_dict.most_common()[0][1] == occurancies_dict.most_common()[1][1]):
      # Call same function again only with a lower K value
      return estimateDataPoint(k - 1, ground_truth, data_point, normalized_data, data_labels, verbose)

  # print debug information
  if verbose:
    print(f" with k: {k}")
    [print(f"Neighbor distance: {round(org_distance[id], 3)} with label: {data_labels[id]} vs ground truth: {ground_truth}")
            for id in nearest_neighbor_ids]
    print(f"Most common neighbor: {occurancies_dict.most_common()[0][0]} in {occurancies_dict}\n")
  
  # Keep track of correct estimations
  if occurancies_dict.most_common()[0][0] == ground_truth: 
    return True
  else: 
    return False

def getKNN(k, normalized_val_data, validation_labels, normalized_data, labels, verbose = False):
  correct = 0
  incorrect = 0
  # Go through all validation data points
  for index, data_point in enumerate(normalized_val_data):
    ground_truth = validation_labels[index]
    
    # print debug information
    if verbose: 
      print(f"Data point number: {index}", end="")

    if estimateDataPoint(k, ground_truth, data_point, normalized_data, labels, verbose):
      correct += 1
    else:
      incorrect += 1

  print(f"Accuracy: {(correct / len(normalized_val_data) * 100)}% with k: {k}")
  return (correct / len(normalized_val_data) * 100)

parser = argparse.ArgumentParser(description='kNearestNeighbors')
parser.add_argument("-k", "--kNN", required=True, type=int, help="amount of neighbors to look at")
parser.add_argument("-v", "--verbose", required=False, default=False, action='store_true', help="enable or disable debug printing")

args = parser.parse_args()

val_data, val_labels, test_data, test_labels = loadNormalizedData()
k = args.kNN
accuracy = getKNN(k, val_data, val_labels, test_data, test_labels, args.verbose)