# Convolutional neural network with TensorFlow to detect sexual images

- This is an academic project, a prototype.

- The main idea is to create a convolutional neural network trained with a few sexual images (~1000 ... finally, these whatsapp groups get a good use) to detect new images.

- The main idea of project is to be used as a moderator helper for a sales website where users can upload their own photographs.

- Flask will be used to POST a new image, analyze it and return a result of such analysis.

- The SFW dataset was downloaded from: http://www.vision.caltech.edu/pmoreels/Datasets/Home_Objects_06/#Download

- As for the NSFW dataset, use your imagination.


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
