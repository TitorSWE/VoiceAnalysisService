import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio 
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

FILE = os.path.join('downloads','recording-c05d7c4a-c34e-4e8f-b659-91fb9999936b.wav')

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

wave = load_wav_16k_mono(FILE)
print("DONE")