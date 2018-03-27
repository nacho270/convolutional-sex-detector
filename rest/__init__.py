from flask import Flask, jsonify, request, redirect,flash


def startFlask():
    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        return 'Hello World!'

    @app.route('/test', methods=['GET'])
    def ping_pong():
        return jsonify({
            'status': 'success',
            'message': 'pong!'
        })

    @app.route('/post', methods=['POST'])
    def postImage():
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