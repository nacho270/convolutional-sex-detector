from flask import Flask, jsonify, request, redirect,flash
from convolution import test_tensorflow

def start_flask():
    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        return 'Hello World!'

    @app.route('/test', methods=['GET'])
    def test():
        return jsonify({
            'status': 'success',
            'message': 'pong!'
        })

    @app.route('/tensor', methods=['GET'])
    def ping_pong():
        return jsonify({
            'status': 'success',
            'message': str(test_tensorflow.run_tf_exampe())
        })

    @app.route('/post', methods=['POST'])
    def post_image():
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        return jsonify({
            'status': 'success',
            'message': 'found file' + file.filename
        })

    app.secret_key = 'nacho'

    app.run()