import tensorflow as tf
import os
import argparse
import json
import logging

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Classifier:

    def __init__(self):

        # Loads labels
        self._label_lines = [line.rstrip() for line in tf.gfile.GFile("tf_files/output_labels.txt")]

        logger.info("Loading model")
        # Load model
        with tf.gfile.FastGFile( "tf_files/output_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        logger.info("Loading final_result tensor")
        # Load softmax tensor
        with tf.Session() as sess:
            self._softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')


    def classify_image(self, image_data):

        logger.info("Classifying image")

        # map to return prediction
        image_prediction = {}

        with tf.Session() as sess:

            # Analize image
            predictions = sess.run(self._softmax_tensor, {'DecodeJpeg/contents:0': image_data})

            # Sort results by probability
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            # Add entry to the dict. label -> probability
            for node_id in top_k:
                image_class = self._label_lines[node_id]
                score = predictions[0][node_id]
                image_prediction[image_class] = str(score)

        return image_prediction


def main(params):
    image_path = params.image_path
    f = open(image_path, 'rb+')
    jpg_data = f.read()
    f.close()
    print(json.dumps(Classifier().classify_image(jpg_data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()
    main(args)
