import tensorflow as tf
import os
import argparse

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def classify_image(image_data):

    # map to return prediction
    image_prediction = {}

    # Loads labels
    label_lines = [line.rstrip() for line in tf.gfile.GFile("~/tf_files/output_labels.txt")]

    # Load model
    with tf.gfile.FastGFile("~/tf_files/output_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:

        # Load softmax tensor
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # Analize image
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort results by probability
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            image_class = label_lines[node_id]
            score = predictions[0][node_id]
            image_prediction[image_class] = str(score)

    return image_prediction


def main(params):
    image_path = params.image_path
    f = open(image_path, 'rb+')
    jpg_data = f.read()
    f.close()
    classify_image(jpg_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()
    main(args)
