# Copyright 2015 The TensorFlow Authors. All Rights Reserved.  #
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
# NOTICE: This work was derived from tensorflow/examples/image_retraining
# and modified to use TensorFlow Hub modules.

# pylint: disable=line-too-long
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

#IMPORTS

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import hashlib
import os.path
import random
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.client import device_lib

#END OF IMPORTS

SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']

FLAGS = None

MIN_NUM_IMAGES_PER_CLASS = 20
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'


FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel')


def detect_devices():
    tf.logging.info("Decting devices")
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    cpus = [x.name for x in local_device_protos if x.device_type == 'CPU']
    if gpus:
        for g in gpus:
            tf.logging.info("GPU {} detected".format(g).upper())
    else:
        tf.logging.info("NO GPU DETECTED, USING CPU")
        for c in cpus:
            tf.logging.info("CPU {} detected".format(c))


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def prepare_file_system():
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)


"""
Reads the image folder and creates a dictionary with the detected labels (key)
and lists of images split by training, testing and validation.
"""
def load_images(image_dir, testing_percentage, validation_percentage):
    tf.logging.info("Loading images from {} using {}% for testing and {}% for validation".format(image_dir, str(testing_percentage) , str(validation_percentage)))

    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None

    # Map variable to return
    result = collections.OrderedDict()

    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))

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

        tf.logging.info("Looking for images in {}".format(dir_name))

        for extension in SUPPORTED_EXTENSIONS:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))

        if not file_list:
            tf.logging.warning('No files found')
            continue

        if len(file_list) < MIN_NUM_IMAGES_PER_CLASS:
            tf.logging.warning('Folder {} has less than {} images, which may cause issues.'.format(dir_name, MIN_NUM_IMAGES_PER_CLASS))
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning('WARNING: Folder {} has more than {} images. Some images will never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))

        training_images = []
        testing_images = []
        validation_images = []
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = (
                (int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) * (100.0 / MAX_NUM_IMAGES_PER_CLASS)
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
        tf.logging.info("Found category " + key)
        value = result[key]
        tf.logging.info("Found {} for training, {} for testing and {} for validation".format(
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
def should_distort_images(flip_left_right, random_crop, random_scale, random_brightness):
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or (random_brightness != 0))


# Download pre-trained graph
def create_module_graph(module_spec):
    height, width = hub.get_expected_image_size(module_spec)

    with tf.Graph().as_default() as graph:
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        m = hub.Module(module_spec)
        bottleneck_tensor = m(resized_input_tensor)
        wants_quantization = any(node.op in FAKE_QUANT_OPS for node in graph.as_graph_def().node)

    return graph, bottleneck_tensor, resized_input_tensor, wants_quantization


'''
Adds a new softmax layer for training in order to retrain the model and identify the specific classes.
Based on: https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
'''
def add_new_layer(class_count, final_tensor_name, bottleneck_tensor, quantize_layer, is_training):

    tf.logging.info("Adding new layer '{}' to be trained for {} classes".format(final_tensor_name, class_count))

    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'Need to specify a batch size'

    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[batch_size, bottleneck_tensor_size], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.int64, [batch_size], name='GroundTruthInput')

    layer_name = 'final_retrain_ops'

    with tf.name_scope(layer_name):

        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    if quantize_layer:
        if is_training:
            tf.contrib.quantize.create_training_graph()
        else:
            tf.contrib.quantize.create_eval_graph()

    tf.summary.histogram('activations', final_tensor)

    # If this is an eval graph, no need to add loss ops or an optimizer.
    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Adds operations that perform JPEG decoding and resizing to the graph..\
def add_jpeg_decoding(module_spec):
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    return jpeg_data, resized_image


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
def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness, module_spec):

    tf.logging.info("Adding distortions. Flip: {}, crop {}%, scale {}%, brigthness {}%".format(flip_left_right, random_crop, random_scale, random_brightness))

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)

    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(shape=[], minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)

    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [input_height, input_width, input_depth])

    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image

    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(shape=[],
                                         minval=brightness_min,
                                         maxval=brightness_max)

    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


'''
From TF doc:
A bottleneck is an informal term used for a layer previous to the final output layer that actually does the classification.
This layer has been trained to output a set of values that's good enough for the classifier to use to distinguish between all the classes it's been asked to recognize.
It has to be a meaningful and compact summary of the images, since it has to contain enough information for the classifier to make a good choice in a very small set of values.

Because every image is reused multiple times during training and calculating each bottleneck takes a significant amount of time,
it speeds things up to cache these bottleneck values on disk so they don't have to be repeatedly recalculated.
If you rerun the script they'll be reused.
'''
def determine_and_cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                                    resized_input_tensor, bottleneck_tensor, module_name):
    tf.logging.info("Determine and cache bottlenecks")

    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)

    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir,
                                         jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)
                how_many_bottlenecks += 1

                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info('{} bottleneck files created for {}.'.format(str(how_many_bottlenecks), category))


''''
Use existing bottleneck or create a new one.
'''
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name):

    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, module_name)

    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor, bottleneck_tensor)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

    error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        error = True

    if error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor, bottleneck_tensor)

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, module_name):
    module_name = (module_name.replace('://', '~')  # URL scheme.
                   .replace('/', '~')  # URL and Unix paths.
                   .replace(':', '~').replace('\\', '~'))  # Windows paths.
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '_' + module_name + '.txt'


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor, bottleneck_tensor):

    tf.logging.info("Creating bottleneck for {} at file  {}".format(label_name, bottleneck_path))

    image_path = get_image_path(image_lists, label_name, index, image_dir, category)

    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    try:
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)

    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    tf.logging.info("Adding evaluation step")
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


"""
If distortions are applied,  recalculate the full model for every image, the cached bottleneck cannot be used.
Get random images for the requested category, run them through the distortion graph, and then the full graph to get the bottleneck results for each.
"""
def get_random_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,distorted_image,
                                     resized_input_tensor, bottleneck_tensor):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []

    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)

        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)

        jpeg_data = tf.gfile.FastGFile(image_path, 'rb').read()
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)

        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)

    return bottlenecks, ground_truths


'''
If no distortions are applied, retrieve the cached bottleneck from disk for images.
Pick a random set of images from the specified category.
'''
def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name):
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

            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category,
                bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)

            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)

                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, module_name)

                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames

# Returns a path to an image for a label at the given index.
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


def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    # Decode the JPEG image.
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    # Run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def save_graph_to_file(graph, graph_file_name, module_spec, class_count):
    sess, _, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [FLAGS.final_tensor_name])

    with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


# Runs a final evaluation on an eval graph using the test data set.
def run_final_eval(train_session, module_spec, class_count, image_lists, jpeg_data_tensor, decoded_image_tensor,
                   resized_image_tensor, bottleneck_tensor):

    tf.logging.info("Running final evaluation")
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(train_session, image_lists, FLAGS.test_batch_size, 'testing', FLAGS.bottleneck_dir,
                                      FLAGS.image_dir, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                                      bottleneck_tensor, FLAGS.tfhub_module))

    (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step, prediction) = build_eval_session(module_spec, class_count)
    test_accuracy, predictions = eval_session.run(
        [evaluation_step, prediction],
        feed_dict={
            bottleneck_input: test_bottlenecks,
            ground_truth_input: test_ground_truth
        }
    )

    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
        tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i]:
                tf.logging.info('%70s  %s' % (test_filename, list(image_lists.keys())[predictions[i]]))


def build_eval_session(module_spec, class_count):

    eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (create_module_graph(module_spec))
    eval_sess = tf.Session(graph=eval_graph)

    with eval_graph.as_default():
        (_, _, bottleneck_input, ground_truth_input, final_tensor) =\
            add_new_layer( class_count, FLAGS.final_tensor_name, bottleneck_tensor, wants_quantization, is_training = False)

        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input, evaluation_step, prediction)


#"Exports model
def export_model(module_spec, class_count, saved_model_dir):
    sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph
    with graph.as_default():
        inputs = {'image': tf.saved_model.utils.build_tensor_info(in_image)}

        out_classes = sess.graph.get_tensor_by_name('final_result:0')
        outputs = {
            'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables( sess, [tf.saved_model.tag_constants.SERVING],
                                                    signature_def_map={ tf.saved_model.signature_constants.
                                                            DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                            signature
                                                    },
                                                    legacy_init_op=legacy_init_op
                                              )
        builder.save()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.image_dir:
        tf.logging.error('Must set flag --image_dir.')
        return -1

    tf.logging.info("TRAINING STARTING")

    detect_devices()

    # Init directories
    prepare_file_system()

    # Load images and validate class count
    image_lists = load_images(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)
    class_count = len(image_lists.keys())

    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
        return -1

    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' + FLAGS.image_dir + ' - multiple classes are needed for classification.')
        return -1

    tf.logging.info("Found {}".format(class_count))

    # Check if distortions should be applied.
    distort_image_enabled = should_distort_images(FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale, FLAGS.random_brightness)
    tf.logging.info("Apply distortions: {}".format(distort_image_enabled))

    # Load pre-trained graph.
    tf.logging.info('Downloading pre-train graph [{}]'.format(FLAGS.tfhub_module))
    module_spec = hub.load_module_spec(FLAGS.tfhub_module)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (create_module_graph(module_spec))

    # Add new layer to train
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) =\
            add_new_layer(class_count, FLAGS.final_tensor_name, bottleneck_tensor, wants_quantization, is_training = True)

    with tf.Session(graph=graph) as sess:

        # Init weights
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up image decoding
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

        if distort_image_enabled:
            # Create distortions
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(FLAGS.flip_left_right,
                                                                                         FLAGS.random_crop,
                                                                                         FLAGS.random_scale,
                                                                                         FLAGS.random_brightness,
                                                                                         module_spec)
        else:
            # Determine and cache bottleneck images
            determine_and_cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir, jpeg_data_tensor,
                                            decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module)

        # Add evaluation the new layer
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        # Write down summaries
        merged = tf.summary.merge_all()

        tf.logging.info("Writing down train summary at {}".format(FLAGS.summaries_dir + '/train'))
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

        tf.logging.info("Writing down validation summary at {}".format(FLAGS.summaries_dir + '/validation'))
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

        train_saver = tf.train.Saver()

        # TRAIN USING THE REQUIRED STEPS QUANTITY
        tf.logging.info("Training using {} steps".format(FLAGS.how_many_training_steps))
        for i in range(FLAGS.how_many_training_steps):
            if distort_image_enabled:
                (train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size,'training', FLAGS.image_dir, distorted_jpeg_data_tensor,
                    distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.train_batch_size, 'training', FLAGS.bottleneck_dir, FLAGS.image_dir,
                    jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, FLAGS.tfhub_module)

            train_summary, _ = sess.run([merged, train_step], feed_dict={bottleneck_input: train_bottlenecks,
                                                                         ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run( [evaluation_step, cross_entropy],
                                                                feed_dict={ bottleneck_input: train_bottlenecks,
                                                                            ground_truth_input: train_ground_truth})

                tf.logging.info('Step %d: Train accuracy = %.1f%%' % (i, train_accuracy * 100))
                tf.logging.info('Step %d: Cross entropy = %f' % (i, cross_entropy_value))

                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks( sess, image_lists, FLAGS.validation_batch_size, 'validation', FLAGS.bottleneck_dir,
                                                   FLAGS.image_dir, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                                                   bottleneck_tensor, FLAGS.tfhub_module))

                # Run a validation step and capture training summaries for TensorBoard
                validation_summary, validation_accuracy = sess.run([merged, evaluation_step],feed_dict={bottleneck_input: validation_bottlenecks,
                                                                                                        ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' % (i, validation_accuracy * 100, len(validation_bottlenecks)))

            # Store intermediate results
            intermediate_frequency = FLAGS.intermediate_store_frequency

            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0):
                # If we want to do an intermediate save, save a checkpoint of the train
                # graph, to restore into the eval graph.
                train_saver.save(sess, CHECKPOINT_NAME)
                intermediate_file_name = (FLAGS.intermediate_output_graphs_dir + 'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' + intermediate_file_name)
                save_graph_to_file(graph, intermediate_file_name, module_spec, class_count)

        # TRAINING COMPLETE
        # Save checkpoint
        train_saver.save(sess, CHECKPOINT_NAME)

        # Run evaluation with some new images not used before.
        tf.logging.info("Training complete. Running evaluation using {} new images".format(FLAGS.test_batch_size))
        run_final_eval(sess, module_spec, class_count, image_lists, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor, bottleneck_tensor)

        # Save model
        if wants_quantization:
            tf.logging.info('The model is instrumented for quantization with TF-Lite')

        tf.logging.info('Writing final model to : ' + FLAGS.output_graph)
        save_graph_to_file(graph, FLAGS.output_graph, module_spec, class_count)

        tf.logging.info("Writing output labels")
        with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        if FLAGS.saved_model_dir:
            export_model(module_spec, class_count, FLAGS.saved_model_dir)

        tf.logging.info("TRAINING FINISHED")


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
    add_arg_param(parser, '--intermediate_output_graphs_dir', str, 'intermediate_output_graphs_dir',
                  'Where to save the intermediate graphs.')
    add_arg_param(parser, '--intermediate_store_frequency', int, 0,
                  'How many steps to store intermediate graph. If "0" then will not store.')
    add_arg_param(parser, '--output_labels', str, 'tf_files/output_labels.txt',
                  'Where to save the trained graph\'s labels.')
    add_arg_param(parser, '--summaries_dir', str, 'tf_files/retrain_logs',
                  'Where to save summary logs for TensorBoard.')
    add_arg_param(parser, '--how_many_training_steps', int, 4000, 'How many training steps to run before ending.')
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
    add_arg_param(parser, '--print_misclassified_test_images', None, False,
                  "Whether to print out a list of all misclassified test images.", 'store_true')
    add_arg_param(parser, '--model_dir', str, '/tmp/imagenet', """\
                                                          Path to classify_image_graph_def.pb,
                                                          imagenet_synset_to_human_label_map.txt, and
                                                          imagenet_2012_challenge_label_map_proto.pbtxt.\
                                                          """)
    add_arg_param(parser, '--bottleneck_dir', str, './tf_files/bottleneck',
                  'Path to cache bottleneck layer values as files.')
    add_arg_param(parser, '--final_tensor_name', str, 'final_result',
                  "The name of the output classification layer in the retrained graph.")
    add_arg_param(parser, '--flip_left_right', None, False,
                  "Whether to randomly flip half of the training images horizontally.", 'store_true')
    add_arg_param(parser, '--random_crop', int, 0,
                  "A percentage determining how much of a margin to randomly crop off the training images.")
    add_arg_param(parser, '--random_scale', int, 0,
                  "A percentage determining how much to randomly scale up the size of the training images by.")
    add_arg_param(parser, '--random_brightness', int, 0,
                  "A percentage determining how much to randomly multiply the training image input pixels up or down by.")
    add_arg_param(parser, '--tfhub_module', str, 'https://tfhub.dev/google/imagenet/inception_v3/classification/1', """\
                                                          Which TensorFlow Hub module to use.
                                                          See https://github.com/tensorflow/hub/blob/master/docs/modules/image.md
                                                          for some publicly available ones.\
                                                          """)
    add_arg_param(parser, '--saved_model_dir', str, '', 'Where to save the exported graph.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
