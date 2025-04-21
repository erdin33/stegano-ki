import os
import uuid
import numpy as np
import cv2
from flask import Flask, render_template, request, send_from_directory, send_file
from PIL import Image
from io import BytesIO


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class DCTSteganography:
    def __init__(self, alpha=6):
        self.alpha = alpha
        self.quant = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])

    def embed_text(self, image_path, text, output_path):
        img = cv2.imread(image_path)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(float)
        h, w = y.shape

        h_pad = 8 - (h % 8) if h % 8 != 0 else 0
        w_pad = 8 - (w % 8) if w % 8 != 0 else 0
        y = np.pad(y, ((0, h_pad), (0, w_pad)), mode='edge')

        binary_text = ''.join(format(ord(c), '08b') for c in text) + '00000000'

        bit_idx = 0
        for i in range(0, y.shape[0] - 7, 8):
            for j in range(0, y.shape[1] - 7, 8):
                if bit_idx >= len(binary_text):
                    break
                block = y[i:i+8, j:j+8]
                dct_block = cv2.dct(block)
                q_block = np.round(dct_block / (self.quant * self.alpha))

                bit = int(binary_text[bit_idx])
                if bit == 1 and q_block[0, 0] % 2 == 0:
                    q_block[0, 0] += 1
                elif bit == 0 and q_block[0, 0] % 2 == 1:
                    q_block[0, 0] -= 1
                q_block[0, 0] = np.clip(q_block[0, 0], -1024, 1023)

                dct_block = q_block * (self.quant * self.alpha)
                y[i:i+8, j:j+8] = cv2.idct(dct_block)
                bit_idx += 1

        y = y[:h, :w]
        ycrcb[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
        stego = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(output_path, stego)

    def extract_text(self, image_path):
        img = cv2.imread(image_path)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(float)

        binary_message = ""
        chars = []

        for i in range(0, y.shape[0] - 7, 8):
            for j in range(0, y.shape[1] - 7, 8):
                block = y[i:i+8, j:j+8]
                dct_block = cv2.dct(block)
                q_block = np.round(dct_block / (self.quant * self.alpha))

                bit = int(q_block[0, 0]) % 2
                binary_message += str(bit)

                if len(binary_message) == 8:
                    if binary_message == "00000000":
                        return ''.join(chars)
                    chars.append(chr(int(binary_message, 2)))
                    binary_message = ""

        return ''.join(chars)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')
        file = request.files.get('image')

        if not file:
            return render_template('index.html', error="Image wajib diisi.")

        filename = str(uuid.uuid4()) + '_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        steg = DCTSteganography()

        if action == 'embed':
            text = request.form.get('message')
            if not text:
                return render_template('index.html', error="Pesan tidak boleh kosong.")

            output_filename = 'stego_' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            steg.embed_text(filepath, text, output_path)

            return render_template(
                'index.html',
                original=filename,
                result=output_filename,
                extracted=None
            )

        elif action == 'extract':
            extracted_text = steg.extract_text(filepath)
            return render_template(
                'index.html',
                original=None,
                result=filename,
                extracted=extracted_text
            )

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/index.html')
def index_html():
    return index()


@app.route('/stegano')
@app.route('/stegano.html')  # Add this second route to handle both URLs
def stegano_page():
    return render_template('stegano.html')

@app.route('/download/<filename>')
def download_file(filename):
    if filename.endswith('.webp'):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with Image.open(image_path) as img:
            img_byte_arr = BytesIO()
            img.convert('RGB').save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return send_file(img_byte_arr, mimetype='image/png', as_attachment=True,
                             download_name=f'{os.path.splitext(filename)[0]}.png')
    else:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)