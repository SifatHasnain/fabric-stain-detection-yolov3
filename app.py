from flask import Flask, request, jsonify
from flask_cors import CORS

from darknet_images import detect_stain


app = Flask(__name__)
CORS(app)


@app.route('/detection/stain', methods=['POST'])
def detect():
    data = request.json['data']

    annotated_image_string, detections = detect_stain(data, save_labels=False, save_image=True)

    response = {
        'defect_confidence': detections,
        'annotated_image': annotated_image_string,
    }

    return jsonify({'data': response}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
