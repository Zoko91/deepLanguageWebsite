from flask import *
import os
from pydub import AudioSegment
import tensorflow as tf
from tensorflow import keras
import librosa, librosa.display,librosa.feature
import numpy as np
import io


app = Flask(__name__)
# New model
path = './static/temp/Models/model3.h5'
model = keras.models.load_model(path)

@app.route("/")
def index():
    current_page = "home"
    return render_template('home.html', current_page=current_page)


@app.route('/save_audio', methods=['POST'])
def save_audio():
    audio_file = request.files['audio']
    audio_data = io.BytesIO(audio_file.read())
    file_path = 'static/temp/recording.webm'
    if os.path.exists(file_path):
        os.remove(file_path)
    sound = AudioSegment.from_file(audio_data, format='webm')
    sound.export('static/temp/recording.wav', format='wav')
    return redirect(url_for("process"))


def preprocessMelCoeff(file_path):
    # Load audio file
    wav, sr = librosa.load(file_path, sr=16000) # load audio file with 16kHz sample rate
    # Pad or truncate the audio file to 5 seconds
    wav = librosa.util.fix_length(wav,size=80000)
    # Calculate Mel-frequency spectrogram
    spect = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power as reference.
    spect = librosa.power_to_db(spect, ref=np.max)
    # Calculate MFCCs from Mel-frequency spectrogram
    mfccs = librosa.feature.mfcc(S=spect, n_mfcc=13)
    return mfccs


def predictLibrosa(file_path):

    # NEW VERSION
    mfccs = preprocessMelCoeff(file_path)
    # convert the numpy array to a tensorflow tensor
    mfccs = tf.convert_to_tensor(mfccs, dtype=tf.float32)
    mfccs = tf.reshape(mfccs, (1, 13, 157, 1))
    prediction = model.predict(mfccs)
    print(prediction)
    # get the index of the predicted class
    language_index = tf.argmax(prediction, axis=1).numpy()[0]
    # define the mapping of class index to language
    language_mapping = {0: 'French', 1: 'Spanish', 2: 'German', 3: 'English'}
    # get the predicted language
    language = language_mapping[language_index]
    # get the probability of the predicted class
    language_probability = prediction[0][language_index]

    return language, language_probability


@app.route('/process')
def process():
    current_page = "Results"
    #to_wav()
    file_path = 'static/temp/recording.wav'
    language, language_probability = predictLibrosa(file_path)
    language_probability = round(language_probability * 100, 2)
    return render_template('process.html', current_page=current_page, language=language, probability=language_probability)

