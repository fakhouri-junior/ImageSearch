import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/salim/caffe/caffe-master/python')

import argparse
import os
import time

from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2
caffe.set_mode_cpu()

'''
Paths to the model, weights, and mean image file
'''
model_deploy = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/deploy.prototxt'
model_weights = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/snapshot_iter_20010.caffemodel'
mean_file_path = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/mean.binaryproto'


def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net
    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)


def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer
    Arguments:
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
        inputs={'data': dims}
    )
    t.set_transpose('data', (2, 0, 1))  # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2, 1, 0))

    if mean_file:
        # set mean pixel
        with open(mean_file, 'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t


def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk
    Returns an np.ndarray (channels x width x height)
    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension
    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image


def forward_pass(images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:, :, np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x + batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        # try
        # print net.blobs['fc7'].data
        # end try
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    return scores


def read_labels(labels_file):
    """
    Returns a list of strings
    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels


def classify(caffemodel, deploy_file, image_files,
             mean_file=None, labels_file=None, batch_size=None, use_gpu=True):
    """
    Classify some images against a Caffe model and print the results
    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images
    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = [load_image(image_file, height, width, mode) for image_file in image_files]
    labels = read_labels(labels_file)

    # Classify the image
    scores = forward_pass(images, net, transformer, batch_size=batch_size)

    ### Process the results
    # print scores
    # print (-scores)
    # minus because argsort gives the smallest index first and so on, by doing minus the biggest becomes the lowest
    # which comes first neat ha
    indices = (-scores).argsort()[:, :5]  # take top 5 results
    # print indices
    classifications = []
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0 * scores[image_index, i], 4)))
        classifications.append(result)

    output = ""
    for index, classification in enumerate(classifications):
        output += '{:-^80}'.format(' Prediction for %s ' % image_files[index])
        output += '\n'
        print '{:-^80}'.format(' Prediction for %s ' % image_files[index])
        for label, confidence in classification:
            output += '{:9.4%} - {}'.format(confidence / 100.0, label)
            output += '\n'
            print '{:9.4%} - "{}"'.format(confidence / 100.0, label)
        print
    return output


def main(caffemodel, deploy_file, image_file, mean_file, labels_file):

    """

    :param caffemodel: path to the model, pretrained one
    :param deploy_file: path to the deply file which holds the information about the model we used
    :param image_file: list of strings, each string is a path to an image saved on the computer
    :param mean_file: path to the mean_file of the images
    :param labels_file: path to the labels.txt
    :return: nothing, just allow classify to print the results.
    """
    script_start_time = time.time()

    result = classify(caffemodel, deploy_file, image_file,
             mean_file, labels_file, use_gpu=False)

    print 'Script took %f seconds.' % (time.time() - script_start_time,)

    return result

def mainWeb(listImagePath):
    """

    :param listImagePath: list of path images
    :return: a string representing the result of the classification
    """
    # make sure listImagePath is a list
    if type(listImagePath) is not list:
        listImagePath = [listImagePath]
    deploy_file = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/deploy.prototxt'
    caffemodel = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/snapshot_iter_20010.caffemodel'
    mean_file = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/mean.binaryproto'
    labels_file = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/formatted_labels.txt'

    script_start_time = time.time()

    result = classify(caffemodel, deploy_file, listImagePath,
                      mean_file, labels_file, use_gpu=False)

    print 'Script took %f seconds.' % (time.time() - script_start_time,)

    return result


"""
CORE RETRIEVAL
"""


# following different approach than lmdb, I will rely on the above functions to pre-process the images
def retrieval(caffemodel, deploy_file, image_files,
              mean_file=None, labels_file=None, batch_size=None, use_gpu=True):
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = [load_image(image_file, height, width, mode) for image_file in image_files]
    labels = read_labels(labels_file)

    f_array = forward_pass_retieval(images, net, transformer, batch_size=batch_size)

    return f_array


def forward_pass_retieval(images, net, transformer, batch_size=None):
    '''
    returns features_array of shape(N(with regards to images),4096)
    '''

    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:, :, np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    features_array = None
    for chunk in [caffe_images[x:x + batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        # try
        fc7_output = net.blobs['fc7'].data
        # end try
        end = time.time()
        if features_array is None:
            features_array = np.copy(fc7_output)
        else:
            features_array = np.vstack((features_array, fc7_output))
        print 'Processed %s/%s images in %f seconds ...' % (len(features_array), len(caffe_images), (end - start))

    return features_array


def main_retrieval(caffemodel, deploy_file, image_file, mean_file):
    script_start_time = time.time()

    result = retrieval(caffemodel, deploy_file, image_file,
                       mean_file, use_gpu=False)

    print 'Script took %f seconds.' % (time.time() - script_start_time,)

    return result
"""
def get_images_list():
    images_list = []
    train_file = open(
        '/home/salim/caffe/caffe-master/myproject/apparel_classification/fashion-data/my_train_full_auto.txt')

    for line in train_file.readlines():
        parts = line.split(' ')
        path = parts[0]
        images_list.append(path)

    return images_list
"""


# let's write a function that takes a path to the test image and find the similar images
from scipy.misc import imread, imshow

def load_array():
    print "I am running"
    my_features_array = np.load('/home/salim/caffe/caffe-master/myproject/features_train_small.npz')
    features_array = my_features_array['arr_0']
    print 'done loading the array'
    return features_array
"""
NEED TO CHANGE THE FEATUREES ARRAY INTO SMALLER ONE, 10 IMAGES PER GATEGORY FOR PERFORMANCE
DONE CHANGED, under features_array_small
"""

def findSimilar(list_path, features_array):
    # first get the image features (fc7)

    # define the params for the main_retrieval
    model_deploy = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/deploy.prototxt'
    model_weights = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/snapshot_iter_20010.caffemodel'
    mean_file_path = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/mean.binaryproto'

    test_features = main_retrieval(model_weights, model_deploy, list_path, mean_file_path)
    # load the dataset features from disk, TAKES TIME !!! WE NEED A BETTER WAY
    #my_features_array = np.load('/home/salim/caffe/caffe-master/myproject/features_train.npz')
    #features_array = my_features_array['arr_0']

    # get the distance matrix
    # Output: sqrt((x-y)^2)
    # (x-y)^2 = x^2 + y^2 - 2xy
    # my distance matrix has the shape (num_test, num_train)
    test_sum = np.sum(np.square(test_features), axis=1)
    train_sum = np.sum(np.square(features_array), axis=1)
    inner_product = np.dot(test_features, features_array.T)
    # make test_num a column vector
    dists = np.sqrt(-2 * inner_product + test_sum.reshape(-1, 1) + train_sum)  # broadcast

    # fetch the images' path
    images_path = images_test

    similarPaths = []
    # stupid approach one by one
    # let's find the indices, argsort gives the index of the lowest value first, which is the closer to the image
    for i in xrange(len(list_path)):
        indices = np.argsort(dists[i, :])
        print indices[:5]
        # use the indices to find the actual image
        # we know that images_path (a list) stores all our dataset images path, they sould match
        for closer in indices[:5]:
            print closer
            path = images_path[closer]
            similarPaths.append(path)
            # let's show the images

            #image = imread(path)
            #imshow(image)

    return similarPaths


images_test = ['/home/salim/Downloads/test/blouse1.jpg', '/home/salim/Downloads/test/blouse2.jpg',
               '/home/salim/Downloads/test/blouse4.jpg', '/home/salim/Downloads/test/blouse3.jpg',
               '/home/salim/Downloads/test/blouse5.jpg', '/home/salim/Downloads/test/blouse6.jpg',
               '/home/salim/Downloads/test/blouse9.jpg', '/home/salim/Downloads/test/blouse7.jpg',
               '/home/salim/Downloads/test/blouse10.jpg', '/home/salim/Downloads/test/blouse8.jpg',

               '/home/salim/Downloads/test/cloak1.jpg', '/home/salim/Downloads/test/cloak2.jpg',
               '/home/salim/Downloads/test/cloak4.jpg', '/home/salim/Downloads/test/cloak3.jpg',
               '/home/salim/Downloads/test/cloak5.jpg', '/home/salim/Downloads/test/cloak6.jpg',
               '/home/salim/Downloads/test/cloak8.jpg', '/home/salim/Downloads/test/cloak7.jpg',
               '/home/salim/Downloads/test/cloak9.jpg', '/home/salim/Downloads/test/cloak10.jpg',

               '/home/salim/Downloads/test/coat1.jpg', '/home/salim/Downloads/test/coat2.jpg',
               '/home/salim/Downloads/test/coat4.jpg', '/home/salim/Downloads/test/coat3.jpg',
               '/home/salim/Downloads/test/coat5.jpg', '/home/salim/Downloads/test/coat6.jpg',
               '/home/salim/Downloads/test/coat9.jpg', '/home/salim/Downloads/test/coat7.jpg',
               '/home/salim/Downloads/test/coat10.jpg', '/home/salim/Downloads/test/coat8.jpg',

               '/home/salim/Downloads/test/jacket1.jpg', '/home/salim/Downloads/test/jacket2.jpg',
               '/home/salim/Downloads/test/jacket10.jpg', '/home/salim/Downloads/test/jacket3.jpg',
               '/home/salim/Downloads/test/jacket9.jpg', '/home/salim/Downloads/test/jacket4.jpg',
               '/home/salim/Downloads/test/jacket8.jpg', '/home/salim/Downloads/test/jacket5.jpg',
               '/home/salim/Downloads/test/jacket7.jpg', '/home/salim/Downloads/test/jacket6.jpg',

               '/home/salim/Downloads/test/long1.jpg', '/home/salim/Downloads/test/long2.jpg',
               '/home/salim/Downloads/test/long10.jpg', '/home/salim/Downloads/test/long3.jpg',
               '/home/salim/Downloads/test/long9.jpg', '/home/salim/Downloads/test/long4.jpg',
               '/home/salim/Downloads/test/long8.jpg', '/home/salim/Downloads/test/long5.jpg',
               '/home/salim/Downloads/test/long7.jpg', '/home/salim/Downloads/test/long6.jpg',

               '/home/salim/Downloads/test/polo1.jpg', '/home/salim/Downloads/test/polo2.jpg',
               '/home/salim/Downloads/test/polo3.jpg',
               '/home/salim/Downloads/test/polo9.jpg', '/home/salim/Downloads/test/polo4.jpg',
               '/home/salim/Downloads/test/polo8.jpg', '/home/salim/Downloads/test/polo5.jpg',
               '/home/salim/Downloads/test/polo7.jpg', '/home/salim/Downloads/test/polo6.jpg',

               '/home/salim/Downloads/test/robe1.jpg', '/home/salim/Downloads/test/robe2.jpg',
               '/home/salim/Downloads/test/robe3.jpg',
               '/home/salim/Downloads/test/robe9.jpg', '/home/salim/Downloads/test/robe4.jpg',
               '/home/salim/Downloads/test/robe8.jpg', '/home/salim/Downloads/test/robe5.jpg',
               '/home/salim/Downloads/test/robe7.jpg', '/home/salim/Downloads/test/robe6.jpg',

               '/home/salim/Downloads/test/shirt1.jpg', '/home/salim/Downloads/test/shirt2.jpg',
               '/home/salim/Downloads/test/shirt10.jpg', '/home/salim/Downloads/test/shirt3.jpg',
               '/home/salim/Downloads/test/shirt9.jpg', '/home/salim/Downloads/test/shirt4.jpg',
               '/home/salim/Downloads/test/shirt8.jpg', '/home/salim/Downloads/test/shirt5.jpg',
               '/home/salim/Downloads/test/shirt7.jpg', '/home/salim/Downloads/test/shirt6.jpg',

               '/home/salim/Downloads/test/short1.jpg', '/home/salim/Downloads/test/short2.jpg',
               '/home/salim/Downloads/test/short10.jpg', '/home/salim/Downloads/test/short3.jpg',
               '/home/salim/Downloads/test/short9.jpg', '/home/salim/Downloads/test/short4.jpg',
               '/home/salim/Downloads/test/short8.jpg', '/home/salim/Downloads/test/short5.jpg',
               '/home/salim/Downloads/test/short7.jpg', '/home/salim/Downloads/test/short6.jpg',

               '/home/salim/Downloads/test/suit1.jpg', '/home/salim/Downloads/test/suit2.jpg',
               '/home/salim/Downloads/test/suit10.jpg', '/home/salim/Downloads/test/suit3.jpg',
               '/home/salim/Downloads/test/suit9.jpg', '/home/salim/Downloads/test/suit4.jpg',
               '/home/salim/Downloads/test/suit8.jpg', '/home/salim/Downloads/test/suit5.jpg',
               '/home/salim/Downloads/test/suit7.jpg', '/home/salim/Downloads/test/suit6.jpg',

               '/home/salim/Downloads/test/sweater1.jpg', '/home/salim/Downloads/test/sweater2.jpg',
               '/home/salim/Downloads/test/sweater10.jpg', '/home/salim/Downloads/test/sweater3.jpg',
               '/home/salim/Downloads/test/sweater9.jpg', '/home/salim/Downloads/test/sweater4.jpg',
               '/home/salim/Downloads/test/sweater8.jpg',
               '/home/salim/Downloads/test/sweater7.jpg', '/home/salim/Downloads/test/sweater6.jpg',

               '/home/salim/Downloads/test/tshirt1.jpg', '/home/salim/Downloads/test/tshirt2.jpg',
               '/home/salim/Downloads/test/tshirt10.jpg', '/home/salim/Downloads/test/tshirt3.jpg',
               '/home/salim/Downloads/test/tshirt9.jpg', '/home/salim/Downloads/test/tshirt4.jpg',
               '/home/salim/Downloads/test/tshirt8.jpg', '/home/salim/Downloads/test/tshirt5.jpg',
               '/home/salim/Downloads/test/tshirt7.jpg', '/home/salim/Downloads/test/tshirt6.jpg',

               '/home/salim/Downloads/test/vest1.jpg', '/home/salim/Downloads/test/vest2.jpg',
               '/home/salim/Downloads/test/vest10.jpg', '/home/salim/Downloads/test/vest3.jpg',
               '/home/salim/Downloads/test/vest9.jpg', '/home/salim/Downloads/test/vest4.jpg',
               '/home/salim/Downloads/test/vest8.jpg', '/home/salim/Downloads/test/vest5.jpg',
               '/home/salim/Downloads/test/vest7.jpg', '/home/salim/Downloads/test/vest6.jpg',

               '/home/salim/Downloads/test/uniform1.jpg', '/home/salim/Downloads/test/uniform2.jpg',
               '/home/salim/Downloads/test/uniform4.jpg', '/home/salim/Downloads/test/uniform3.jpg']



# BELOW IS JUST FOR TESTING

#image_path = ['/home/salim/Downloads/suit1.jpg','/home/salim/Downloads/suit2.jpg','/home/salim/Downloads/suit3.jpg',
 #            '/home/salim/Downloads/short1.jpg', '/home/salim/Downloads/under1.jpg','/home/salim/Downloads/under2.jpg']
#labels_path = '/home/salim/caffe/caffe-master/myproject/mymodel-all-finetune/formatted_labels.txt'
#result = main(model_weights, model_deploy, image_path, mean_file_path, labels_path)


