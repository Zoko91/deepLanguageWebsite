from flask import *
import os
from pydub import AudioSegment
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
import pyaudio
import wave
import librosa, librosa.display,librosa.feature
import numpy as np


app = Flask(__name__)
# New model
# model = keras.models.load_model('/Users/josephbeasse/Desktop/deepLanguage/Models/model3.h5')
model = keras.models.load_model('/Users/josephbeasse/Desktop/deepLanguage/workingDirectory/Models/model.h5')

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

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess(file_path):
    wav = load_wav_16k_mono(file_path)
    wav = wav[16000:96000]
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    wav = wav / tf.math.reduce_max(wav)
    return wav

def extract_mfccs(file_path):
    preprocessed_audio = preprocess(file_path)
    # if not tf.reduce_any(tf.math.is_finite(preprocessed_audio)):
    #     print("Detected NaN values")
    #     tf.print(file_path)
    # Get the audio data as a tensor
    audio_tensor = tf.convert_to_tensor(preprocessed_audio)
    # Reshape the audio data to 2D for the STFT function
    audio_tensor = tf.reshape(audio_tensor, (1, -1))
    # Perform STFT on the audio data
    stft = tf.signal.stft(audio_tensor, frame_length=2048, frame_step=512)
    # Get the magnitude of the complex STFT output
    magnitude = tf.abs(stft)
    # Apply a logarithm to the magnitude to get the log-magnitude
    log_magnitude = tf.math.log(magnitude + 1e-9)
    # Apply a Mel filter bank to the log-magnitude to get the Mel-frequency spectrum
    mel_spectrum = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=40,
                                                         num_spectrogram_bins=magnitude.shape[-1],
                                                         sample_rate=16000,
                                                         lower_edge_hertz=20,
                                                         upper_edge_hertz=8000)
    mel_spectrum = tf.tensordot(log_magnitude, mel_spectrum, 1)
    # Perform the DCT to get the MFCCs
    mfccs = tf.signal.dct(mel_spectrum, type=2, axis=-1, norm='ortho')
    # Get the first 13 MFCCs, which are the most important for speech recognition
    mfccs = mfccs[..., :13]
    # Normalize the MFCCs
    mfccs = (mfccs - tf.math.reduce_mean(mfccs)) / tf.math.reduce_std(mfccs)
    return mfccs

def to_wav():
    file_name = 'static/temp/recording.mp3'
    # Load the MP3 file using the from_mp3() method
    mp3_audio = AudioSegment.from_mp3(file_name)
    output_name = 'static/temp/recording.wav'
    # Save the audio as a WAV file using the export() method
    mp3_audio.export(output_name, format="wav")




def predict(file_path):

    # NEW VERSION
    mfccs = extract_mfccs(file_path)
    mfccs = tf.expand_dims(mfccs, axis=0)
    prediction = model.predict(mfccs)
    # get the index of the predicted class
    language_index = tf.argmax(prediction, axis=2).numpy()[0][0]
    # define the mapping of class index to language
    language_mapping = {0: 'English', 1: 'French', 2: 'German', 3: 'Spanish'}
    # get the predicted language
    language = language_mapping[language_index]
    # get the probability of the predicted class
    language_probability = prediction[0][0][language_index]

    return language, language_probability


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
    language_mapping = {0: 'English', 1: 'French', 2: 'German', 3: 'Spanish'}
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
    RECORD_SECONDS = 6
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
