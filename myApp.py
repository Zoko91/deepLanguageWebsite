from flask import *
import os
from pydub import AudioSegment
import tensorflow as tf
from tensorflow import keras
import pyaudio
import wave
import librosa, librosa.display,librosa.feature
import numpy as np


app = Flask(__name__)
# New model
model = keras.models.load_model('/Users/josephbeasse/Desktop/deepLanguage/Models/__largeModels/model3.h5')

@app.route("/")
def index():
    current_page = "home"
    return render_template('home.html', current_page=current_page)

@app.route('/upload', methods=['GET','POST'])
def upload():
    file = request.files.get('file')
    file_path = 'static/temp/' + file.filename
    if os.path.exists(file_path):
        os.remove(file_path)
    # Save the file to the server
    file.save(file_path)
    return redirect('/process')

def to_wav():
    file_name = 'static/temp/recording.mp3'
    # Load the MP3 file using the from_mp3() method
    mp3_audio = AudioSegment.from_mp3(file_name)
    output_name = 'static/temp/recording.wav'
    # Save the audio as a WAV file using the export() method
    mp3_audio.export(output_name, format="wav")


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

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 32000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "static/temp/recording.wav"

    if os.path.exists(WAVE_OUTPUT_FILENAME):
        os.remove(WAVE_OUTPUT_FILENAME)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


@app.route('/process')
def process():
    current_page = "Results"
    #to_wav()
    file_path = 'static/temp/recording.wav'
    language, language_probability = predictLibrosa(file_path)
    language_probability = round(language_probability * 100, 2)
    return render_template('process.html', current_page=current_page, language=language, probability=language_probability)

# A UTILISER QUAND ON APPUIE SUR LE BOUTON
@app.route("/record", methods=["GET","POST"])
def record():
    # Call the record_audio function here
    record_audio()
    return redirect(url_for("process"))
