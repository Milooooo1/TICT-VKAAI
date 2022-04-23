import numpy as np
import random
import argparse

verbose = False
num_iterations = 0

def importCSVFile(filename, year):
  data = np.genfromtxt(filename, delimiter=";", usecols=[1,2,3,4,5,6,7], 
                      converters={5: lambda s: 0 if s == b"-1" else float(s), 
                                  7: lambda s: 0 if s == b"-1" else float(s)}
                      )

  dates = np.genfromtxt(filename, delimiter=";", usecols=[0])

  if year == 2000:
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
        
  else:
    labels = []
    for label in dates:
      if label < 20010301:
        labels.append("winter")
      elif 20010301 <= label < 20010601:
        labels.append("lente")
      elif 20010601 <= label < 20010901:
        labels.append("zomer")
      elif 20010901 <= label < 20011201:
        labels.append("herfst")
      else: # from 01-12 to end of year
        labels.append("winter")


  return labels, data

def loadNormalizedData():
  test_labels, test_data = importCSVFile("dataset1.csv", 2000)
  validation_labels, validation_data = importCSVFile("validation1.csv", 2001)

  norm_max = max([np.amax(test_data), np.amax(validation_data)])
  norm_min = min([np.amin(test_data), np.amin(validation_data)])

  # Normalize validation and test data
  normalized_test_data = np.array([[((i - norm_min) / (norm_max - norm_min)) for i in day] for day in test_data])
  normalized_val_data  = np.array([[((i - norm_min) / (norm_max - norm_min)) for i in day] for day in validation_data])

  return normalized_val_data, validation_labels, normalized_test_data, test_labels

def adjustCentroidToAvg(cluster_dict):
    new_cluster_dict = {}

    # Go through every cluster
    for cluster_num in cluster_dict.keys():
		
        # If a centroid has no cluster we cannot average its position
        if (len(cluster_dict[cluster_num][1]) == 0):
            new_cluster_dict[cluster_num] = [cluster_dict[cluster_num][0], []]    
            continue

        # Create an empty list the same size of a data_point
        cluster_point_average = [0] * len(cluster_dict[cluster_num][1][0])
        # For every cluster go through all cluster points
        for cluster_point in cluster_dict[cluster_num][1]:
            # For the cluster point go through all its features and add them up
            for index, feature in enumerate(cluster_point):
                cluster_point_average[index] += feature
        
        new_centroid = [cluster_point_average[index] / len(cluster_dict[cluster_num][1]) for index in range(0, len(cluster_point_average))]
        new_cluster_dict[cluster_num] = [new_centroid, []]

    return new_cluster_dict

def getCentroids(norm_test_data, cluster_dict, iteration = 0):
    global verbose

    centroids = np.array([cluster_dict[cluster][0] for cluster in cluster_dict.keys()])
    for data_point in norm_test_data:
        # Get distance of the data_point to all the centroids
        distances = [np.linalg.norm(centroid - data_point) for centroid in centroids]
    
        # Add the datapoint to the list of the centroid with the smallest distance
        cluster_dict[distances.index(min(distances))][1].append(data_point)

    # Adjust all centroids to their respective new center
    new_cluster_dict = adjustCentroidToAvg(cluster_dict)
    
    # Get all new centroids and compare them to the old ones if they did not change we are done 
    new_centroids = np.array([np.array(new_cluster_dict[cluster][0]) for cluster in new_cluster_dict.keys()])
    if (centroids == new_centroids).all():
        if verbose:
            print(f"Final centroids after: {iteration} adjustments")
            [print(f"\tCluster {index} : centroid : {[str(round(item, 3)) for item in centroid]}") for index, centroid in enumerate(new_centroids)]
        return [new_cluster_dict[centroid][0] for centroid in new_cluster_dict.keys()]

    # Caculate the clusters again based on the new centroids
    return getCentroids(norm_test_data, new_cluster_dict, iteration+1)

#TODO doe dit spelletje 5x opnieuw aangezien we beginnen met random punten. Pak als resultaat de beste iteratie.
def getBestCentroids(norm_test_data, num_clusters): 
    global num_iterations

    results = []
    for iteration in range(0, num_iterations):
        centroids = []
        prev = -1
        for i in range(0, num_clusters):

            # Pick a random already existing point 
            # but avoid picking the same point twice
            pick = random.randint(0, len(norm_test_data))
            while pick == prev:
                pick = random.randint(0, len(norm_test_data))
                prev = pick
            
            # Add centroids to list
            centroids.append(norm_test_data[pick])
        
        # Initialize the cluster dict by creating a dictionary with keys for the amount of clusters
        # and the values being a mutable list of the centroid and its cluster of data_points
        cluster_dict = {index : [centroid, []] for index, centroid in enumerate(centroids)}
        results.append(getCentroids(norm_test_data, cluster_dict))

    best_centroids = []
    # if verbose:
        # print(f"After {num_iterations} iterations the best centroids are:")
        # [print(f"\tFor cluster {index}: ") for index, centroid in results]

    return cluster_dict

def createScreePlot():
    # kwadrateer de afstand tussen alle punten van het cluster ten opzichte van de centroid = aggregraat interclusterdistance 
    pass

def main():
    random.seed(1)
    global verbose 
    global num_iterations

    parser = argparse.ArgumentParser(description='kNearestNeighbors')
    parser.add_argument("-i", "--iterations", required=True, type=int, help="amount of times to adjust the centroids")
    parser.add_argument("-v", "--verbose", required=False, default=False, action='store_true', help="enable or disable debug printing")

    args = parser.parse_args()
    verbose = args.verbose
    num_iterations = args.iterations

    norm_val_data, val_labels, norm_test_data, test_labels = loadNormalizedData()
    centroids = getBestCentroids(norm_test_data, 4)
    # [print(f"Centroid: {centroid}") for centroid in centroids]

if __name__ == '__main__':
    main()