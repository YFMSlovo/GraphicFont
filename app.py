from flask import Flask, request, send_file
import generate
import os

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate_image():
    image_file = request.files['image']
    text = request.form.get('text')
    image_path = 'temp_image.jpg'
    image_file.save(image_path)
    generate.generate_image(image_path, text)
    generated_image_path = f"generated_{text}.png"
    return send_file(generated_image_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
