# Convolutional neural network with TensorFlow to detect sexual images

- This is an academic project, a prototype.

- The main idea is to create a convolutional neural network trained with a few sexual images (~1000 ... finally, these whatsapp groups get a good use) to detect new images.

- The main idea of project is to be used as a moderator helper for a sales website where users can upload their own photographs.

- Flask will be used to POST a new image, analyze it and return a result of such analysis.

- The SFW dataset was downloaded from: http://www.vision.caltech.edu/pmoreels/Datasets/Home_Objects_06/#Download

- As for the NSFW dataset, use your imagination.

BTW, there's a `cats_and_dogs` branch which i used to hand-in the proyect since the professor didn't allow use the real one when it was about an actual problem in real life when a user uploads a photo on a public website.

## Train

To train the model:

`cd convolutional-sex-detector/convolution`
`python3 train.py --image_dir images --bottleneck_dir tf_files/bottleneck --output_labels tf_files/output_labels.txt --output_graph tf_files/output_graph.pb --summaries_dir tf_files/retrain_logs`

To run tensorboard:

`tensorboard --logdir tf_files/retrain_logs`

## Run

There are 2 options to run this app
    
1 - Through the console:
`python3 convolution/image_classifier.py --image_path "image_path.jpg"`

2 - Through the REST api: `python3 main.py`
 
 Then
 
 `curl -X POST -F "image=@image_path.JPG" localhost:5000/classify`

## Monitor

`tensorboard --logdir tf_files/retrain_logs`


## Parms & HELP

```
python3 convolution/train.py --help
usage: train.py [-h] [--image_dir IMAGE_DIR] [--output_graph OUTPUT_GRAPH]
                [--output_labels OUTPUT_LABELS]
                [--summaries_dir SUMMARIES_DIR]
                [--training_steps TRAINING_STEPS]
                [--learning_rate LEARNING_RATE]
                [--testing_percentage TESTING_PERCENTAGE]
                [--validation_percentage VALIDATION_PERCENTAGE]
                [--eval_step_interval EVAL_STEP_INTERVAL]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--test_batch_size TEST_BATCH_SIZE]
                [--validation_batch_size VALIDATION_BATCH_SIZE]
                [--print_misclassified_test_images] [--model_dir MODEL_DIR]
                [--bottleneck_dir BOTTLENECK_DIR]
                [--final_tensor_name FINAL_TENSOR_NAME] [--flip_left_right]
                [--random_crop RANDOM_CROP] [--random_scale RANDOM_SCALE]
                [--random_brightness RANDOM_BRIGHTNESS]

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Path to folders of labeled images.
  --output_graph OUTPUT_GRAPH
                        Where to save the trained graph.
  --output_labels OUTPUT_LABELS
                        Where to save the trained graph's labels.
  --summaries_dir SUMMARIES_DIR
                        Where to save summary logs for TensorBoard.
  --training_steps TRAINING_STEPS
                        How many training steps to run before ending.
  --learning_rate LEARNING_RATE
                        How large a learning rate to use when training.
  --testing_percentage TESTING_PERCENTAGE
                        What percentage of images to use as a test set.
  --validation_percentage VALIDATION_PERCENTAGE
                        What percentage of images to use as a validation set.
  --eval_step_interval EVAL_STEP_INTERVAL
                        How often to evaluate the training results.
  --train_batch_size TRAIN_BATCH_SIZE
                        How many images to train on at a time.
  --test_batch_size TEST_BATCH_SIZE
                        How many images to test on. This test set is only used
                        once, to evaluate the final accuracy of the model
                        after training completes. A value of -1 causes the
                        entire test set to be used, which leads to more stable
                        results across runs.
  --validation_batch_size VALIDATION_BATCH_SIZE
                        How many images to use in an evaluation batch. This
                        validation set is used much more often than the test
                        set, and is an early indicator of how accurate the
                        model is during training. A value of -1 causes the
                        entire validation set to be used, which leads to more
                        stable results across training iterations, but may be
                        slower on large training sets.
  --print_misclassified_test_images
                        Whether to print out a list of all misclassified test
                        images.
  --model_dir MODEL_DIR
                        Path to classify_image_graph_def.pb,
                        imagenet_synset_to_human_label_map.txt, and
                        imagenet_2012_challenge_label_map_proto.pbtxt.
  --bottleneck_dir BOTTLENECK_DIR
                        Path to cache bottleneck layer values as files.
  --final_tensor_name FINAL_TENSOR_NAME
                        The name of the output classification layer in the
                        retrained graph.
  --flip_left_right     Whether to randomly flip half of the training images
                        horizontally.
  --random_crop RANDOM_CROP
                        A percentage determining how much of a margin to
                        randomly crop off the training images.
  --random_scale RANDOM_SCALE
                        A percentage determining how much to randomly scale up
                        the size of the training images by.
  --random_brightness RANDOM_BRIGHTNESS
                        A percentage determining how much to randomly multiply
                        the training image input pixels up or down by.

```

## Dependencies

- Python 3.6 or higher
- Tensorflow 1.8.0
- Flask 0.12.2
- Tensorboard (to monitor training) `pip3 install tensorboard`


If you want to find out which version of the libraries you have installed, run in your console:

```
python3 -c 'import tensorflow as tf; print(tf.__version__)'
python3 -c 'import flask as flask; print(flask.__version__)' 
```

## Tensorflow nstallation instructions
- https://www.tensorflow.org/install/install_linux
- https://www.tensorflow.org/install/install_mac
- https://www.tensorflow.org/install/install_windows
