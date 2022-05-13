#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display as ipd

def volume(debussy_file):

 ipd.Audio(debussy_file)

# load audio files with librosa
 debussy, sr = librosa.load(debussy_file)
 FRAME_SIZE = 1024
 HOP_LENGTH = 512

 sc_debussy = librosa.feature.spectral_centroid(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
 sc_debussy.shape
 frames = range(len(sc_debussy))
 t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
 len(t)
 plt.figure(figsize=(25,7))
 # line colour is red
 plt.axhline(y = 5000, color = 'b', linestyle = 'dashed',label = "threshold of normal volume")  
 plt.plot(t, sc_debussy, color='r',label = "Intensity of sound")
 plt.xlabel('Time (sec)')
 plt.ylabel('vloume')
 # plotting the legend
 # Add label
 plt.legend(loc = 'upper left')
 plt.savefig('imags/volume of voice.png')
 plt.show()
      





