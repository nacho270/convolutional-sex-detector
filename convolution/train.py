# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Image learning network based on Inception v3 architecture model.

This model adds a new top layer to Inception model (based on ImageNet images)
than recognize sexual images to be used in websites moderation modules.

Bare in mind that is an academic model and might no scale to heavy traffic without a proper bigdata architecture.

The top layer receives as input a 2048-dimensional array for each image.
Then, the a softmax layer is trained on top.

Folder organization example:

- photos/NSFW/nsfw_photo1.jpg
- photos/NSFW/nsfw_photo2.jpg
- photos/SFW/sfw_photo1.jpg
- photos/SFW/sfw_photo2.jpg

The subfolders indicate the label (class) of each image. File names are not important.

Execution example

```bash
python convolutional-sex-detector/convolution/retrain.py --image_dir ~/photos
```

Sorry for the method documentation format, i don't like python standard xD
It's hard to read and annoying in the middle of the method.

"""

# IMPORTS
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
import logging

import numpy as np
import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

# END OF IMPORTS

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
script_dir = os.path.dirname(__file__)

# GLOBAL FLAGS FOR TF
FLAGS = None

INCEPTION_V3_DOWNLOAD_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# Parameters used by Inception V3
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
MIN_NUM_IMAGES_PER_CLASS = 20


# Downloads the inception v3 model.
def download_inception_v3():
    logger.info("Attempting to download inception v3 model")
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = INCEPTION_V3_DOWNLOAD_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # CLOSURE
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(INCEPTION_V3_DOWNLOAD_URL,
                                                 filepath,
                                                 _progress)
        print()
        statinfo = os.stat(filepath)
        logger.info('Successfully downloaded', filepath, statinfo.st_size, 'bytes.')
    else:
        logger.info("Inception v3 model already present")
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


# Creates a graph with the Inception V3 model, the bottleneck, jpeg_data and resize input tensors
def create_inception_graph():
    logger.info("Loading inception graph and creating bottleneck_tensor, jpeg_data and resized_input tensors")
    with tf.Graph().as_default() as inception_graph:
        model_filename = os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))
    return inception_graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


"""
Reads the image folder and creates a dictionary with the detected labels (key)
and lists of images split by training, testing and validation.
"""
def load_images(image_dir, testing_percentage, validation_percentage):
    logger.info("Loading images from {} using {}% for testing and {}% for validation".format(image_dir, str(testing_percentage) , str(validation_percentage)))

    if not gfile.Exists(image_dir):
        logger.info("Image directory '" + image_dir + "' not found.")
        return None

    # Map variable to return
    result = {}

    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            # Skip root directory
            is_root_dir = False
            continue

        file_list = []
        dir_name = os.path.basename(sub_dir)

        if dir_name == image_dir:
            continue

        logger.info("Looking for images in {}".format(dir_name))
        for extension in SUPPORTED_EXTENSIONS:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))

        if not file_list:
            logger.info('No files found')
            continue

        if len(file_list) < MIN_NUM_IMAGES_PER_CLASS:
            logger.warning('Folder {} has less than {} images, which may cause issues.'.format(dir_name, MIN_NUM_IMAGES_PER_CLASS))
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            logger.info('WARNING: Folder {} has more than {} images. Some images will '
                        'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))

        training_images = []
        testing_images = []
        validation_images = []
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = (
                (int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) *  (100.0 / MAX_NUM_IMAGES_PER_CLASS)
            )

            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # ADD MAP ENTRY
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }

    # LOG
    for key in result:
        logger.info("Found category " + key)
        value = result[key]
        logger.info("Found {} for training, {} for testing and {} for validation".format(
            str(len(value["training"])),
            str(len(value["testing"])),
            str(len(value["validation"]))
        ))

    return result


'''
Checks if image distortions need to be applied using:

- flip_left_right: Boolean whether to randomly mirror images horizontally.
- random_crop: Integer percentage setting the total margin used around the crop box.
- random_scale: Integer percentage of how much to vary the scale by.
- random_brightness: Integer range to randomly multiply the pixel values by.
'''
def distort_images(flip_left_right, random_crop, random_scale, random_brightness):
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or (random_brightness != 0))

'''
From TF doc:
A bottleneck is an informal term used for a layer previous to the final output layer that actually does the classification.
This layer has been trained to output a set of values that's good enough for the classifier to use to distinguish between all the classes it's been asked to recognize.
It has to be a meaningful and compact summary of the images, since it has to contain enough information for the classifier to make a good choice in a very small set of values.

Because every image is reused multiple times during training and calculating each bottleneck takes a significant amount of time,
it speeds things up to cache these bottleneck values on disk so they don't have to be repeatedly recalculated.
If you rerun the script they'll be reused.
'''
def determine_and_cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
    logging.info("Determine and cache bottlenecks")
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    logger.info('{} bottleneck files created for {}.'.format(str(how_many_bottlenecks), category))


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


'''
Use existing bottleneck or create a new one.
'''
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category)

    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

    error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        logger.info('Invalid float found, recreating bottleneck')
        error = True

    if error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'


def get_image_path(image_lists, label_name, index, image_dir, category):
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)

    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)

    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)

    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor):
    logger.info("Creating bottleneck for {} at file {}".format(label_name, bottleneck_path))
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)

    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)

    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    except:
        raise RuntimeError('Error during processing file %s' % image_path)

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


# Runs inference on an image to extract the bottleneck summary.
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

'''
Adds a new softmax layer for training in order to retrain the model to identify the specific classes.
Based on: https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
'''
def add_new_layer(class_count, final_tensor_name, bottleneck_tensor):
    logger.info("Adding new layer '{}' to be trained for {} classes".format(final_tensor_name, class_count))

    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default( bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):

        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth_input, logits=logits)

        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    logger.info("Adding evaluation step")
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal( prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


'''
If no distortions are applied, retrieve the cached bottleneck from disk for images.
Pick a random set of images from the specified category.
'''
def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir, image_dir, jpeg_data_tensor, bottleneck_tensor):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []

    if how_many >= 0:
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                  image_index, image_dir, category,
                                                  bottleneck_dir, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                      image_index, image_dir, category,
                                                      bottleneck_dir, jpeg_data_tensor,
                                                      bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


"""
If distortions are applied,  recalculate the full model for every image, the cached bottleneck cannot be used.
Get random images for the requested category, run them through the distortion graph, and then the full graph to get the bottleneck results for each.
"""
def get_random_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir, input_jpeg_tensor, distorted_image, resized_input_tensor, bottleneck_tensor):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []

    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)

        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)

        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                             resized_input_tensor,
                                             bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


""" Construct a network of operations to apply them to an image.

    If the distortions are to be applied, apply crops, scales and flips to support real world variations and, thus, have a more effective model.

    Cropping

    Place a box at a random position in the full image. The cropping parameter controls the size of that box in relation to the input image.
    - If it's zero, then the box is the same size as the input.
    - If the value is 50%, then the crop box will be half the width and height of the input.

    Scaling

    Like cropping, but the box is always centered and its size varies randomly within the given range.
    - If the scale percentage is zero, the box is the same size as the input.
    - If If it's 50%, then the box will be in a random range between half the width and height and full size.

    Params:
      - flip_left_right: Boolean whether to randomly mirror images horizontally.
      - random_crop: Integer percentage setting the total margin used around the crop box.
      - random_scale: Integer percentage of how much to vary the scale by.
      - random_brightness: Integer range to randomly multiply the pixel values by graph.
    """
def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness):
    logger.info("Adding distortions. Flip: {}, crop {}%, scale {}%, brigthness {}%".format(flip_left_right, random_crop, random_scale, random_brightness))
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)

    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)

    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH])

    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image

    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # Set up the pre-trained graph.
    download_inception_v3()

    inception_graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())
    image_lists = load_images(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)
    class_count = len(image_lists.keys())

    if class_count == 0:
        logger.info('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        logger.info('Only one valid folder of images found at ' + FLAGS.image_dir +
                    ' - multiple classes are needed for classification.')
        return -1

    # Check if distortions should be applied.
    distort_image_enabled = distort_images(FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale, FLAGS.random_brightness)
    logger.info("Apply distortions: {}".format(distort_image_enabled))

    with tf.Session(graph = inception_graph) as sess:

        if distort_image_enabled:
            # Create distortions
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(FLAGS.flip_left_right,
                                                                                         FLAGS.random_crop,
                                                                                         FLAGS.random_scale,
                                                                                         FLAGS.random_brightness)
        else:
            # Determine and cache bottleneck images
            determine_and_cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)

        # Add new layer to train
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_new_layer(
                len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_tensor)

        # Add evaluation the new layer
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

        # Write down summaries
        merged = tf.summary.merge_all()

        logger.info("Writing down train summary at {}".format(FLAGS.summaries_dir + '/train'))
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

        logger.info("Writing down validation summary at {}".format(FLAGS.summaries_dir + '/validation'))
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

        #Init weights
        sess.run(tf.global_variables_initializer())

        # TRAIN USING THE REQUIRED STEPS QUANTITY
        logger.info("Training using {} steps".format(FLAGS.training_steps))
        for i in range(FLAGS.training_steps):
            if distort_image_enabled:
                (train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.image_dir, distorted_jpeg_data_tensor,
                    distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    bottleneck_tensor)

            train_summary, _ = sess.run([merged, train_step], feed_dict={bottleneck_input: train_bottlenecks,
                                                                         ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            is_last_step = (i + 1 == FLAGS.training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks,
                                                                                                             ground_truth_input: train_ground_truth})

                logger.info('Step %d: Train accuracy = %.1f%%' % (i, train_accuracy * 100))
                logger.info('Step %d: Cross entropy = %f' % (i, cross_entropy_value))

                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks(sess, image_lists, FLAGS.validation_batch_size, 'validation',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor, bottleneck_tensor))

                validation_summary, validation_accuracy = sess.run([merged, evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                logger.info('Step %d: Validation accuracy = %.1f%% (N=%d)' % (i, validation_accuracy * 100, len(validation_bottlenecks)))

        # TRAINING COMPLETE
        # Run evaluation with some new images not used before.
        logger.info("Training complete. Running evaluation using {} new images".format(FLAGS.test_batch_size))
        test_bottlenecks, test_ground_truth, test_filenames = (
            get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size, 'testing', FLAGS.bottleneck_dir,
                                          FLAGS.image_dir, jpeg_data_tensor, bottleneck_tensor))

        test_accuracy, predictions = sess.run( [evaluation_step, prediction], feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        logger.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        if FLAGS.print_misclassified_test_images:
            logger.info('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    logger.info('%70s  %s' % (test_filename, list(image_lists.keys())[predictions[i]]))

        # Write out the trained graph and labels with the weights stored as constants.

        logger.info("Writing final model")
        output_graph_def = graph_util.convert_variables_to_constants(sess, inception_graph.as_graph_def(), [FLAGS.final_tensor_name])

        with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

    logger.info("FINISHED")


def add_arg_param(parser, option, type, default, help, action=None):
    if action is not None:
        if type is not None:
            parser.add_argument(
                option,
                type=type,
                default=default,
                help=help,
                action=action
            )
        else:
            parser.add_argument(
                option,
                default=default,
                help=help,
                action=action
            )
    else:
        if type is not None:
            parser.add_argument(
                option,
                type=type,
                default=default,
                help=help
            )
        else:
            parser.add_argument(
                option,
                default=default,
                help=help
            )


if __name__ == '__main__':
    # COMMAND LINE PARAMETERS PARSER
    parser = argparse.ArgumentParser()
    add_arg_param(parser, '--image_dir', str, '', 'Path to folders of labeled images.')
    add_arg_param(parser, '--output_graph', str, 'tf_files/output_graph.pb', 'Where to save the trained graph.')
    add_arg_param(parser, '--output_labels', str, 'tf_files/output_labels.txt', 'Where to save the trained graph\'s labels.')
    add_arg_param(parser, '--summaries_dir', str, 'tf_files/retrain_logs', 'Where to save summary logs for TensorBoard.')
    add_arg_param(parser, '--training_steps', int, 4000, 'How many training steps to run before ending.')
    add_arg_param(parser, '--learning_rate', float, 0.01, 'How large a learning rate to use when training.')
    add_arg_param(parser, '--testing_percentage', int, 10, 'What percentage of images to use as a test set.')
    add_arg_param(parser, '--validation_percentage', int, 10, 'What percentage of images to use as a validation set.')
    add_arg_param(parser, '--eval_step_interval', int, 10, 'How often to evaluate the training results.')
    add_arg_param(parser, '--train_batch_size', int, 100, 'How many images to train on at a time.')
    add_arg_param(parser, '--test_batch_size', int, -1, """\
                                                      How many images to test on. This test set is only used once, to evaluate
                                                      the final accuracy of the model after training completes.
                                                      A value of -1 causes the entire test set to be used, which leads to more
                                                      stable results across runs.\
                                                      """)
    add_arg_param(parser, '--validation_batch_size', int, 100, """\
                                                      How many images to use in an evaluation batch. This validation set is
                                                      used much more often than the test set, and is an early indicator of how
                                                      accurate the model is during training.
                                                      A value of -1 causes the entire validation set to be used, which leads to
                                                      more stable results across training iterations, but may be slower on large
                                                      training sets.\
                                                      """)
    add_arg_param(parser, '--print_misclassified_test_images', None, False, "Whether to print out a list of all misclassified test images.", 'store_true')
    add_arg_param(parser, '--model_dir', str, '/tmp/imagenet', """\
                                                      Path to classify_image_graph_def.pb,
                                                      imagenet_synset_to_human_label_map.txt, and
                                                      imagenet_2012_challenge_label_map_proto.pbtxt.\
                                                      """)
    add_arg_param(parser, '--bottleneck_dir', str, './tf_files/bottleneck', 'Path to cache bottleneck layer values as files.')
    add_arg_param(parser, '--final_tensor_name', str, 'final_result', "The name of the output classification layer in the retrained graph.")
    add_arg_param(parser, '--flip_left_right', None, False, "Whether to randomly flip half of the training images horizontally.", 'store_true')
    add_arg_param(parser, '--random_crop', int, 0, "A percentage determining how much of a margin to randomly crop off the training images.")
    add_arg_param(parser, '--random_scale', int, 0, "A percentage determining how much to randomly scale up the size of the training images by.")
    add_arg_param(parser, '--random_brightness', int, 0, "A percentage determining how much to randomly multiply the training image input pixels up or down by.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
