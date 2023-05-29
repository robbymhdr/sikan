# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,
    LeakyReLU, Input, GlobalAveragePooling2D
)
from PIL import Image
from fungsi import make_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as ts

# =[Variabel Global]=============================

def PredGambar(file_gmbr):
    file = file_gmbr
    gmbr_array = np.asarray(file)
    gmbr_array = gmbr_array*(1/225)
    gmbr_input = tf.reshape(gmbr_array, shape=[1, 150, 150, 3])

    predik_array = model.predict(gmbr_input)[0]

    df = pd.DataFrame(predik_array)
    df = df.rename({0: 'NilaiKemiripan'}, axis='columns')
    Kualitas = ['Ikan Segar', 'Ikan Tidak Segar']
    df['Kelas'] = Kualitas
    df = df[['Kelas', 'NilaiKemiripan']]

    predik_kelas = np.argmax(model.predict(gmbr_input))

    if predik_kelas == 0:
        predik_Kualitas = 'Ikan Segar'
    else:
        predik_Kualitas = 'Ikan Tidak Segar'

    return predik_Kualitas, df

# =[Variabel Global]=============================


app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 22500 * 22500
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.JPG']
app.config['UPLOAD_PATH'] = './static/images/uploads/'

# model = None

NUM_CLASSES = 2
cifar10_classes = ["Ikan Segar", "Ikan tidak segar"]

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]


@app.route("/")
def beranda():
    return render_template('index.html')

@app.route("/beranda")
def beranda_2():
    return render_template('index.html')

# [Routing untuk API]


@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi
    hasil_prediksi = '(none)'
    gambar_prediksi = '(none)'

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)

    # Periksa apakah ada file yg dipilih untuk diupload
    if filename != '':

        # Set/mendapatkan extension dan path dari file yg diupload
        file_ext = os.path.splitext(filename)[1]
        gambar_prediksi = '/static/images/uploads/' + filename

        # Periksa apakah extension file yg diupload sesuai (jpg)
        if file_ext in app.config['UPLOAD_EXTENSIONS']:

            # Simpan Gambar
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_PATH'], filename))

            # Memuat Gambar
            lok = '.' + gambar_prediksi
            
            gmbr = load_img(lok, target_size=(150, 150))
            
            x = img_to_array(gmbr)
            x = np.expand_dims(x, axis=0)
            gmbr = np.vstack([x])

            # Prediksi Gambar
            kelas, df = PredGambar(gmbr)
            hasil_prediksi = kelas

            # Return hasil prediksi dengan format JSON
            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })
        else:
            # Return hasil prediksi dengan format JSON
            gambar_prediksi = '(none)'
            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })

# =[Main]========================================		

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = make_model()
	model.load_weights("fish_classification_model.h5")

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)