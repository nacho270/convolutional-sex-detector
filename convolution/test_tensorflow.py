import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

def run_tf_exampe() -> str:
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    return sess.run(hello)