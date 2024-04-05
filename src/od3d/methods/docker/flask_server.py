# pip install Flask
# FLASK_ENV=development FLASK_APP=flask_server.py flask run

from flask import Flask, jsonify, request
import pickle
app = Flask(__name__)
import io
import torch

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if not file:
            return "No selected file"

        file_bytes = request.files['file'].read()
        bytes_io = io.BytesIO(file_bytes)
        data = pickle.load(bytes_io)
        print(data)
        data_return = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data_return[key] = value.tolist()
            else:
                data_return[key] = value
        #data_return['status_code'] = 200
        #data_return['message'] = 'OK'
        return jsonify(data_return)


if __name__ == '__main__':
    app.run()