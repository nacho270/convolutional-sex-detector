from flask import Flask, jsonify, request, redirect
from convolution import Classifier

def start_flask():

    app = Flask(__name__)

    classifier = Classifier()

    @app.route('/ping', methods=['GET'])
    def test():
        return jsonify({
            'status': 'success',
            'message': 'pong!'
        })

    @app.route('/classify', methods=['POST'])
    def classify_image():
        if 'image' not in request.files:
            return redirect(request.url)
        image = request.files.get('image', '')
        return jsonify(classifier.classify_image(image.read()))


    app.run()