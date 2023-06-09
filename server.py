from flask import Flask, jsonify, request
import os
import tensorflow as tf
import tensorflow_io as tfio 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from pydub import AudioSegment
import ffmpeg
import subprocess

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase/isboozed-firebase-adminsdk-bnfo5-e7d3c65002.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'isboozed.appspot.com'
})

# Create a storage client
bucket = storage.bucket()


def convert(file_name):
    # Run FFmpeg command
    input_file = "downloads/" + file_name
    name_without_extension = file_name.split('.')[0]
    print("WHITOUT EXT", name_without_extension)
    output_file = "download_wav/" + name_without_extension + ".wav"
    subprocess.run(['ffmpeg', '-i', input_file, output_file])


def download(file_name):

    # Download the file
    blob = bucket.blob('audio/'+file_name)
    blob.download_to_filename('downloads/'+file_name)

    print('File downloaded successfully.')

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    print("READING", file_contents)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess(file_path): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


# Import trained CNN
model = tf.keras.models.load_model('models')

app = Flask(__name__)

@app.route("/<file_name>")
def is_angry(file_name):
    
    # Downloading the file
    download(file_name)

    # Converting into wav
    convert(file_name)
    wav_file_name = file_name.split('.')[0] + ".wav"

    # Spectrogramm
    file_path = os.path.join('download_wav',wav_file_name)
    spectrogram = preprocess(file_path)
    print("spectrogram ok")
    spectrogram = tf.reshape(spectrogram, (-1, 1491, 257, 1))
    print("reshape ok")

    # Predicting
    yhat = model.predict(spectrogram)
    print("predict ok")
    print(yhat[0][0])
    
    if (yhat[0][0]>0.5):
        return "happy"
    return "angry" 

if __name__ == '__main__':
    app.run()