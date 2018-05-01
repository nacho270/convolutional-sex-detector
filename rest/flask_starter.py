from flask import Flask, jsonify, request, redirect
from convolution import image_classifier

def start_flask():

    app = Flask(__name__)

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
        return jsonify(image_classifier.classify_image(image.read()))


    app.run()