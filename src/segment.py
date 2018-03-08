'''
Author:        Alexander Zhu
Date Created:  5 March, 2018
Description:   Implementation of K-means++ for image segmentation.
'''

from __future__ import print_function

import sys
import numpy as np
from matplotlib import pyplot as plt
import time
import getopt
import logging
from PIL import Image
from collections import OrderedDict
import itertools

# Global logger
logger = logging.getLogger('ImageSegmentator')

class SegmentatorException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

def squared_euclidean_distance(x, y):
    ''' Compute the squared Euclidean distance between two points.

    @param {1D float numpy array} x - a point
    @param {1D float numpy array} y - a point
    @returns {float} the squared Euclidean distance
    '''
    return np.linalg.norm(np.subtract(x, y)) ** 2

def objective_function(original, segmented):
    ''' Compute the overall loss.

    @param {1D numpy array} original - the original flattened image data
    @param {1D numpy array} clustered - the segmented flattened image data
    @returns {float} the result of the objective function
    '''
    cost = sum(squared_euclidean_distance(x,y) for x,y in itertools.izip(original, segmented))
    return cost

def compute_squared_distances(means, data):
    ''' For each data point x, compute the distance between x and the nearest mean.

    @param {set} means - set of selected means
    @param {1D numpy array} data - the flattened image data
    @returns {1D float numpy array} array of distances, one per pixel
    @returns {float} sum of all distances (for weighted sampling)
    '''
    distances = []
    distances_sum = 0

    for i in range(data.shape[0]):
        min_dist = float('inf')
        for mean in means:
            dist = squared_euclidean_distance(data[i], mean)
            min_dist = min(min_dist, dist)

        distances.append(min_dist)
        distances_sum += min_dist

    return np.array(distances), distances_sum

def weighted_random(weights, weights_sum):
    ''' Choose a random item from a weighted distribution.

    @param {1D float numpy array} weights - array of weights forming the distribution
    @param {float} weights_sum - sum of all weights
    @returns {tuple} selected indices
    '''
    # Generate a random number: [0, weights_sum)
    r = np.random.rand() * weights_sum
    count = 0

    for i in range(weights.shape[0]):
        count += weights[i]
        if r < count:
            return i

    raise SegmentatorException('Input image did not have at least k distinct pixels!')

def k_means_plus_plus_initialization(data, k):
    ''' Initialize the means according to K-means++

    @param {1D numpy array} data - the flattened image data
    @param {integer} k - number of means
    @returns {1D numpy array} initialized means
    '''
    logger.debug('Using K-means++ initialization.')

    # Choose initial center randomly
    selected_means = set()
    logger.debug('Selecting mean 1.')
    selected_means.add(tuple(data[np.random.randint(0, data.shape[0])]))
    
    # Choose the remaining k-1 initial means according to K-means++
    for i in range(k-1):
        logger.debug('Selecting mean {0}.'.format(i+2))
        distances, distances_sum = compute_squared_distances(selected_means, data)
        new_mean_idx = weighted_random(distances, distances_sum)

        if tuple(data[new_mean_idx]) in selected_means:
            # We should be selecting unique initial means
            raise SegmentatorException('Got repeated selected mean!')

        selected_means.add(tuple(data[new_mean_idx]))

    means = np.array(list(selected_means), dtype='float64')
    return means

def k_means_initialization(data, k):
    ''' Randomly initialize the means, according to K-means.

    @param {1D numpy array} data - the flattened image data
    @param {integer} k - number of means
    @returns {1D numpy array} initialized means
    '''
    max_pixel_value = np.amax(data)
    mean_shape = (k, data[0].size)
    means = np.multiply(np.random.rand(*mean_shape), kMax_pixel_value)
    return means

def segment(data, k, mean_init=k_means_plus_plus_initialization, distance=squared_euclidean_distance):
    ''' Segment the image with K-means.

    @param {1D numpy array} data - the flattened image data
    @param {integer} k - number of means
    @param {function} mean_init - function to use for initializing the means
    @param {function} distance - function to use for measuring distance between two points
    @returns {1D numpy array} segmented image data
    '''
    start = time.time()
    logger.info('Segmenting image into {0} segments.'.format(k))
    
    
    means = mean_init(data, k)  # Initialize means
    logger.debug('Initial means: {0}.'.format(means))
    clusters = np.zeros(data.shape[0], dtype='uint8')  # Initialize all clusters as zero
    cost = None  # Track the costs over time

    threshold = 10
    num_reassigned = threshold + 1  # Arbitrary number > threshold
    while num_reassigned > threshold:
        clusters, means, num_reassigned, _ = lloyd_iteration(logger, data, clusters, means, distance)
        new_image = np.array([means[clusters[i]] for i in range(data.shape[0])], dtype='uint8')

        # Debug output
        logger.debug('Num reassigned: {0}.'.format(num_reassigned))
        new_cost = objective_function(data, new_image)
        logger.debug('New cost: {0}'.format(new_cost))
        if cost:
            logger.debug('Delta: {0}'.format(new_cost - cost))
        cost = new_cost

    total_time = int(time.time() - start)
    logger.info('Segmenting took {0} seconds.'.format(total_time))

    return new_image

def map_reduce_lloyd_iteration(logger, data, clusters, means, distance):
    # Pass in set of means, data (split per mapper)
    # Mappers compute where each point should go, also keeps running sum of totals for new mean
    # When all are done, pass arguments to reducers and start
    # Reducers compute new centroid by looking at totals, combines list of points
    def map(data, clusters, means, distance):
        pass

    def combine():
        pass

    def reduce():
        pass

    def map_reduce_run(logger, data, clusters, means, distance):
        pass

def lloyd_iteration(logger, data, clusters, means, distance):
    ''' One step of Lloyd iteration.
    First determine clusters by assign each pixel to the closest mean.
    Then recompute the mean of each cluster.

    @param {1D numpy array} data - the flattened image data
    @param {1D numpy array} clusters - the current clusters
    @param {1D numpy array} means - the current means
    @param {function} distance - function to use for measuring distance between two points
    @returns {1D numpy array} new clusters
    @returns {1D numpy array} new means
    @returns {integer} number of reassigned points
    @returns {OrderedDict} dictionary mapping old mean to [new mean, [point indices]]
    '''
    num_reassigned = 0
    # Map mean to [new mean, [point indices]]
    cluster_dict = OrderedDict([tuple(mean), [np.zeros(means[0].shape), []]] for mean in means)

    for i in range(data.shape[0]):
        x = data[i]
        closest_mean_idx = 0
        min_dist = float('inf')
        for k in range(means.shape[0]):
            mean = means[k]
            dist = distance(x, mean)
            if dist < min_dist:
                closest_mean_idx = k
                min_dist = dist

        if clusters[i] != closest_mean_idx:
            num_reassigned += 1
            clusters[i] = closest_mean_idx  # Reassign
        
        # Update dictionary
        closest_mean = means[closest_mean_idx]
        cluster_dict[tuple(closest_mean)][0] += x
        cluster_dict[tuple(closest_mean)][1].append(i)

    # Compute new means
    for i, key in enumerate(cluster_dict):
        if cluster_dict[key][1]:
            cluster_dict[key][0] /= len(cluster_dict[key][1])
            means[i] = cluster_dict[key][0]
        else:
            means[i] = np.zeros(means[0].shape)

    return clusters, means, num_reassigned, cluster_dict

def load_image(image_path):
    ''' Load and return data as a numpy array.

    @param {string} image_path - path to image
    @returns {PIL Image} loaded image
    '''
    try:
        img = Image.open(image_path)
    except IOError as err:
        raise SegmentatorException(str(err))

    logger.info('Loaded image: {0}.'.format(image_path))
    logger.debug('Image shape: {0}.'.format(img.size))
    return img

def save_image(data, image_path):
    ''' Save numpy array as image.

    @param {3D uint8 numpy array} data - the image data
    @param {string} image_path - path to image
    '''
    img = Image.fromarray(data)
    logger.info('Saving image: {0}.'.format(image_path))
    img.save(image_path)
    logger.info('Saved successfully!')

def init_logger(debug):
    ''' Initialize logger.

    @param {boolean} debug - show debug output if True
    @returns {Logger} initialized logger
    '''
    level = logging.DEBUG if debug else logging.INFO
    format_string = '%(asctime)s %(levelname)s %(message)s' if debug else '%(message)s'

    logger = logging.getLogger('ImageSegmentator')
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter(format_string)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Initialized logger.')
    logger.debug('Debug output enabled.')
    return logger

def intro():
    ''' Print intro text. '''
    print('-----------------------------------------------------------------------')
    print('                           Image Segmentator                           ')
    print('                             Version 1.1.0                             ')
    print('                         Author: Alexander Zhu                         ')
    print('-----------------------------------------------------------------------')
    print('')

def usage():
    ''' Print usage information for ImageSegmentator '''
    print('-----------------------------------------------------------------------')
    print('usage: segment.py [-p IMAGE_PATH] [-k K] [-m MAX_SIZE] [-n N] [-h] [-v]')
    print('')
    print('required arguments:')
    print(' -p, --path IMAGE_PATH  path to the image to segment')
    print('')
    print('optional arguments:')
    print(' -k K (default 5)                       number of clusters')
    print(' -m, --max_size MAX_SIZE (default 256)  max scaled image side length')
    print(' -n, --num_trials N (default 1)         number of times to segment')
    print(' -h, --help                             show this help message and exit')
    print(' -d, --debug (default False)            enable debug output')
    print('-----------------------------------------------------------------------')

def getopts():
    ''' Process and return command line arguments. '''
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'p:k:m:n:hd', ['path=', 'max_size=', 'num_trials=', 'help', 'debug'])
    except getopt.GetoptError as err:
        print(err, file=sys.stderr)
        usage()
        sys.exit(2)

    debug = False
    k = 5
    max_size = 256
    n = 1
    image_path = None
    for o, a in opts:
        if o in ('-h', '--help'):
            usage()
            sys.exit()
        elif o in ('-d', '--debug'):
            debug = True
        elif o in ('-k', '--k'):
            try:
                k = int(a)
            except (TypeError, ValueError):
                usage()
                sys.exit()
        elif o in ('-m', '--max_size'):
            try:
                max_size = int(a)
            except (TypeError, ValueError):
                usage()
                sys.exit()
        elif o in ('-n', '--num_trials'):
            try:
                n = int(a)
            except (TypeError, ValueError):
                usage()
                sys.exit()
        elif o in ('-p', '--path'):
            image_path = a
        else:
            assert False, 'unhandled option'

    if image_path == None:
        usage()
        sys.exit()

    return debug, k, max_size, n, image_path

if __name__ == '__main__':
    intro()

    debug, k, max_size, n, image_path = getopts()
    init_logger(debug)

    # Load image
    original_img = load_image(image_path)
    original_shape = original_img.size
    original_data = np.array(original_img, dtype='uint8')[...,:3]
    
    # Compress image
    logger.info('Compressing image.')
    original_img.thumbnail((max_size, max_size), Image.ANTIALIAS)
    compressed_data = np.array(original_img, dtype='uint8')[...,:3]  # RGB
    compressed_shape = compressed_data.shape
    logger.debug('Compressed image shape: {0}.'.format(compressed_shape))
    
    # Flatten into array of RGB tuples
    flattened = np.reshape(compressed_data, (compressed_data.shape[0] * compressed_data.shape[1], -1))
    logger.debug('Flattened image shape: {0}'.format(flattened.shape))

    # Run n times
    for i in range(n):
        logger.info('Trial {0}.'.format(i+1))
        # Segment image
        segmented_data = segment(flattened, k)

        # Unflatten
        unflattened = np.reshape(segmented_data, compressed_shape)
        
        # Enlarge image
        logger.info('Re-enlarging image.')
        enlarged_img = Image.fromarray(unflattened)
        enlarged_img = enlarged_img.resize(original_shape, Image.ANTIALIAS)
        enlarged_data = np.array(enlarged_img, dtype='uint8')[...,:3]

        cost = objective_function(original_data, enlarged_data)
        logger.debug('Cost: {0}.'.format(cost))
        
        save_image(enlarged_data, '{0}_k{1}_{2}.png'.format(image_path, k, int(cost)))
