import numpy as np
import random
import argparse
import collections
import matplotlib.pyplot as plt
import tqdm

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

def getBestCentroids(norm_test_data, num_clusters): 
    global num_iterations

    results = []
    for iteration in range(0, num_iterations):
        centroids = []
        prev = -1
        for i in range(0, num_clusters):

            # Pick a random already existing point 
            # but avoid picking the same point twice
            pick = random.randint(0, len(norm_test_data)-1)
            while pick == prev:
                pick = random.randint(0, len(norm_test_data)-1)
                prev = pick
            
            # Add centroids to list
            centroids.append(norm_test_data[pick])
        
        # Initialize the cluster dict by creating a dictionary with keys for the amount of clusters
        # and the values being a mutable list of the centroid and its cluster of data_points
        cluster_dict = {index : [centroid, []] for index, centroid in enumerate(centroids)}
        results.append(getCentroids(norm_test_data, cluster_dict))

    # Calculate the best iteration by looking for the lowest total distance
    iteration_distances = []
    for iteration in results:

        cluster_dict = {index : 0 for index in range(0, len(iteration))}  
        for data_point in norm_test_data:
            # Find the most near centroid and add that distance to the cluster total dict
            distances = [np.linalg.norm(centroid - data_point) for centroid in iteration]
            cluster_dict[distances.index(min(distances))] += min(distances)
        
        iteration_distances.append(sum([cluster_dict[index] for index in cluster_dict.keys()]))

    if verbose: print(f"Best iteration is iteration: {iteration_distances.index(min(iteration_distances))} with a total distance of {min(iteration_distances)}")
    return results[iteration_distances.index(min(iteration_distances))]

def createScreePlot(norm_test_data):
    testedKparams = range(1, 15)

    screePlotPoints = []
    print("Generating screePlot")
    for k in tqdm.tqdm(testedKparams):
        centroids = getBestCentroids(norm_test_data, k)

        # Initialise a cluster dict, the cluster index as the key and a value 
        # with a mutable list of the centroid and the list of its datapoint
        cluster_dict = {index : [centroid, []] for index, centroid in enumerate(centroids)}
        for data_point in norm_test_data:
            # Get distance of the data_point to all the centroids
            distances = [np.linalg.norm(centroid - data_point) for centroid in centroids]
        
            # Add the datapoint to the list of the centroid with the smallest distance
            cluster_dict[distances.index(min(distances))][1].append(data_point)
   
        # Calculate the intra cluster distance
        # This is the distance between the centroid and all of its data_points
        intraclusterdist = []
        for index, centroid in enumerate(centroids):
            cluster_total = 0
            for cluster_point in cluster_dict[index][1]:
                cluster_total += pow(np.linalg.norm(centroid - cluster_point), 2)
            intraclusterdist.append(cluster_total)

        screePlotPoints.append(sum(intraclusterdist))

    plt.plot(testedKparams, screePlotPoints)
    plt.title("Screeplot")
    plt.xlabel("Amount of clusters")
    plt.xticks(np.arange(min(testedKparams), max(testedKparams)+1, 1.0))
    plt.ylabel("Intra cluster distance")
    plt.savefig("screeplot.png")

def labelCentroids(data, labels, centroids):

    cluster_dict = {index : [centroid, []] for index, centroid in enumerate(centroids)}
    for index, data_point in enumerate(data):
        # Get distance of the data_point to all the centroids
        distances = [np.linalg.norm(centroid - data_point) for centroid in centroids]

        # Add the datapoint to the list of the centroid with the smallest distance
        cluster_dict[distances.index(min(distances))][1].append(labels[index])
    
    res = []
    for cluster in cluster_dict.keys():
        res.append((cluster_dict[cluster][0], collections.Counter(cluster_dict[cluster][1]).most_common()[0][0]))
    
    return res

def measureAccuracy(val_data, val_labels, labeled_centroids):
    correct = 0
    incorrect = 0
    for index, val_point in enumerate(val_data):
        distances = [np.linalg.norm(centroid[0] - val_point) for centroid in labeled_centroids]
        closest_centroid = labeled_centroids[distances.index(min(distances))]
        if closest_centroid[1] == val_labels[index]:
            correct += 1
        else:
            incorrect += 1
    
    print(f"kMeans accuracy is {(correct / len(val_labels) * 100)}%")

def main():
    random.seed(0)
    global verbose 
    global num_iterations

    parser = argparse.ArgumentParser(description='kNearestNeighbors')
    parser.add_argument("-i", "--iterations", required=False, default=5, type=int, 
                                             help="amount of times find the best centroids")
    parser.add_argument("-v", "--verbose", required=False, default=False, 
                        action='store_true', help="enable or disable debug printing")
    parser.add_argument("-c", "--createScreePlot", required=False, default=False, 
                        action='store_true', help="enable or disable scree plot creation")
    parser.add_argument("-m", "--measureAccuracy", required=False, default=True, 
                        action='store_true', help="enable or disable kMeans accuracy measurement")

    args = parser.parse_args()
    verbose = args.verbose
    num_iterations = args.iterations

    norm_val_data, val_labels, norm_test_data, test_labels = loadNormalizedData()
    
    if args.createScreePlot:
        createScreePlot(norm_test_data)
        print("Screeplot saved to filesystem")
    
    if args.measureAccuracy:
        centroids = getBestCentroids(norm_test_data, 3)
        labeled_centroids = labelCentroids(norm_test_data, test_labels, centroids)
        if verbose: print(); [print(f"Centroid: {[str(round(i, 2)) for i in item[0]]} with label: {item[1]}") for item in labeled_centroids]; print()
        measureAccuracy(norm_val_data, val_labels, labeled_centroids)

if __name__ == '__main__':
    main()