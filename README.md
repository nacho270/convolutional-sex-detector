# Convolutional neural network with TensorFlow to detect sexual images

This is an academic project, a prototype.

The main idea is to create a convolutional neural network trained with a few sexual images (~1000 ... finally, these whatsapp groups get a good use) to detect new images.

Flask will be used to POST a new image, analyze it and return a result of such analysis.

Bear in mind that **I am NOT a Python developer**, and therefore the code might not be as elegant as a proper Python developer would expect or to be configured properly (sorry, I learned during the process).

Feel free to fork this project and continue to carry on with the development and please, make it elegant (not like me, though I tried my best).

I developed this using PyCharm as IDE, just in case you wonder

## Dependencies

- Python 3.6 or higher
- Tensorflow 1.8.0
- Flask 0.12.2
- Tensorflow Hub (`pip3 install tensorflow-hub`)
- On mac, run `/Applications/Python 3.6/Install Certificates.command`

If you want to find out which version of the libraries you have installed, run in your console:

```
python3 -c 'import tensorflow as tf; print(tf.__version__)'
python3 -c 'import flask as flask; print(flask.__version__)' 
```

## Installation instructions
- https://www.tensorflow.org/install/install_linux
- https://www.tensorflow.org/install/install_mac
- https://www.tensorflow.org/install/install_windows
