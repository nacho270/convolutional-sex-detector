# Convolutional neural network with TensorFlow to detect sexual images

This is an academic project, a prototype.

The main idea is to create a convolutional neural network trained with a few sexual images (~1000 ... finally, these whatsapp groups get a good use) to detect new images. The main idea of project is to be used as a moderator helper for a sales website where users can upload their own photographs.

Flask will be used to POST a new image, analyze it and return a result of such analysis.

The SFW dataset was downloaded from: http://www.vision.caltech.edu/pmoreels/Datasets/Home_Objects_06/#Download

As for the NSFW dataset, use your imagination.

## Dependencies

- Python 3.6 or higher
- Tensorflow 1.8.0
- Flask 0.12.2
- Tensorflow Hub (`pip3 install tensorflow-hub`)
- On mac, run `/Applications/Python 3.6/Install Certificates.command` to install the necessary certificates to download the tensorflow-hub models.

If you want to find out which version of the libraries you have installed, run in your console:

```
python3 -c 'import tensorflow as tf; print(tf.__version__)'
python3 -c 'import flask as flask; print(flask.__version__)' 
```

## Installation instructions
- https://www.tensorflow.org/install/install_linux
- https://www.tensorflow.org/install/install_mac
- https://www.tensorflow.org/install/install_windows
