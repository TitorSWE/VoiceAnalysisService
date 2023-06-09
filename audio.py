import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio 
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

YES_FILE = os.path.join('audio','yes','Enregistrement-_30_.wav')
NO_FILE  = os.path.join('audio','yes','Enregistrement-_32_.wav')

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

wave = load_wav_16k_mono(YES_FILE)
print("-----LOADING DONE-----")

# CREATE TENSORFLOW DATASET

POS = os.path.join('audio', 'yes')
NEG = os.path.join('audio', 'no')

pos = tf.data.Dataset.list_files(POS+'/*.wav')
neg = tf.data.Dataset.list_files(NEG+'/*.wav')

# LABEL
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

print("-----LABELLING DONE-----")

# Determine Average Length of a positive audio

# Calculate Wave Cycle Length
lengths = []
for file in os.listdir(os.path.join('audio', 'yes')):
    tensor_wave = load_wav_16k_mono(os.path.join('audio', 'yes', file))
    lengths.append(len(tensor_wave))

# Calculate Mean, Min and Max
tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)

print("-----CALCULUS DONE-----")

# Build Preprocessing Function to Convert to Spectrogram

# ATTENTION METTRE LABEL EN OPTION
def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

# Test spectrogram

# Get the first positive audio
filepath, label = positives.as_numpy_iterator().next()
print(filepath)
spectrogram, label = preprocess(filepath, label)
print("-----SPECTROGRAM BUILD-----")

# Verifying plt is working 
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()
print("-----PLOT-----")

# Create Training and Testing Partitions

# pipeline
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(1)
data = data.prefetch(1)

# Split into Training and Testing Partitions
train = data.take(10)
test = data.skip(10).take(4)

samples, labels = train.as_numpy_iterator().next()
print("-----SPLITTING DONE-----")

#Build Sequential Model, Compile and View Summary

model = Sequential()
model.add(Conv2D(4, (3,3), activation='relu', input_shape=(1491, 257,1)))
model.add(Conv2D(4, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
print(model.summary())


hist = model.fit(train, epochs=4, validation_data=test)
print("-----FITTING-----")

yhat = model.predict(samples)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
print(yhat)
print(labels)
print("-----OK-----")

model.save('models')
print("-----SAVING DONE-----")