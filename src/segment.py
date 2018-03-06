from __future__ import print_function

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from PIL import Image
'''
# Load image
img = Image.open('test.png')
data = np.array(img, dtype='uint8')

# Save as npy
np.save('test.npy', data)
img_array = np.load('test.npy')

# Show image
plt.imshow(data)
plt.show()

# Save image twice
matplotlib.image.imsave('new_test.png', img_array)
img = Image.fromarray(data)
img.save('other_new_test.png')'''

def squared_euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def objective_function(data, clusters, means):
    cost = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            cost += squared_euclidean_distance(data[i][j], means[clusters[i][j]])

    return cost

def segment(data, k, distance=squared_euclidean_distance):
    ''' Segment the image with K-means '''
    kMax_pixel_value = 255
    kNum_channels = 3
    mean_shape = (k, kNum_channels)

    print('Segmenting data of shape', data.shape)
    
    # Randomly initialize means
    means = np.multiply(np.random.rand(*mean_shape), kMax_pixel_value)
    #means = np.array([[0, 255, 0], [255, 0, 0]])
    print('Initial means', means)

    # Initialize all clusters as zero
    clusters = np.zeros(data.shape[:2], dtype='uint8')
    #print('Clusters shape', clusters.shape)

    num_reassigned = 1
    while num_reassigned > 0:
        clusters, num_reassigned = assign_clusters(data, clusters, means, distance)
        print('num_reassigned', num_reassigned)

        means = compute_means(data, clusters, mean_shape)
        #print('New means:', means)

    new_image = np.zeros(data.shape, dtype='uint8')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_image[i][j] = means[clusters[i][j]]

    return new_image, clusters, means


def assign_clusters(data, clusters, means, distance):
    num_reassigned = 0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = data[i][j]
            closest_mean = 0
            min_dist = float('inf')
            for k in range(means.shape[0]):
                mean = means[k]
                dist = distance(x, mean)
                if dist < min_dist:
                    closest_mean = k
                    min_dist = dist

            if clusters[i][j] != closest_mean:
                clusters[i][j] = closest_mean  # reassign
                num_reassigned += 1

    #print('Means:', means)
    #print('Clusters:', clusters)
    return clusters, num_reassigned

def compute_means(data, clusters, mean_shape):
    means = np.zeros(mean_shape)

    # Initialize dictionary of empty lists
    split_clusters = {}
    for i in range(mean_shape[0]):
        split_clusters[i] = []

    # Populate dictionary, mapping cluster id to list of data indices
    for i in range(clusters.shape[0]):
        for j in range(clusters.shape[1]):
            split_clusters[clusters[i][j]].append((i, j))

    for cluster_id, indices in split_clusters.items():
        # Protect against clusters of size zero
        if len(indices) == 0:
            means[cluster_id] = 0
            continue

        sum = np.zeros(mean_shape[1])
        for index in indices:
            sum += data[index[0]][index[1]]

        means[cluster_id] = np.divide(sum, len(indices))

    return means

def load_image(image_name):
    ''' Load and return data as a numpy array '''
    img = Image.open(image_name)
    data = np.array(img, dtype='uint8')[...,:3]  # RGB
    return data

def save_image(data, image_name):
    ''' Save numpy array as image '''
    img = Image.fromarray(data)
    img.save(image_name)
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Run: python ImageSegmentator image_name.png')
        sys.exit(0)

    image_name = sys.argv[1]
    data = load_image(image_name)
    #data = np.array([[[20, 0, 0], [0, 20, 0], [10, 10, 0], [225, 225, 0], [200, 250, 0], [250, 200, 0]]])
    #print('Data:', data)
    k = 8
    num_trials = 3
    for i in range(num_trials):
        segmented_data, clusters, means = segment(data, k)
        cost = objective_function(segmented_data, clusters, means)
        print('Cost:', cost)
        save_image(segmented_data, '%s_k%d_%d.png' %(sys.argv[1], k, int(cost)))