import torch
import torchaudio
from torchaudio import transforms
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from random import shuffle
from math import floor
from pydub import AudioSegment
import os
import math, random
import matplotlib.pyplot as plt

# DataLoader class to extract 5 different kinds of Audio Features
# 1. MFCC -> Built on MSS so no need 
# 2. Chromagram
# 3. Mel-Scaled Spectrogram
# 4. Spectral Contrast
# 5. Tonal Centroid Features
#
# 1 Dataloader class per 1 audio sample

class DataLoader_extraction():
    def __init__(self, path = None, normalize = True, sr = 16000):
        if path != None:
            self.path = path

            sound = AudioSegment.from_wav(self.path)
            sound = sound.set_channels(1)
            os.remove(self.path)
            sound.export(self.path, format="wav")

            y = torchaudio.load(path, normalize = normalize)
            transform = transforms.Resample(y[1], 16000)
            y = transform(y[0])
            self.y = [y, 16000]
            self.sr = self.y[1]
            self.ynumpy = (self.y[0][0]).numpy()
        else:
            self.ynumpy = None
            self.y = None
            self.sr = sr

    def rechannel_resample(self, channel, sr):

        self.rechannel(channel)
        self.resample(sr)

        return

    def rechannel(self, new_channel):
        sig, sr = self.y
        if (sig.shape[0] == new_channel):
            return self.y
        if (new_channel == 1):
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])

        self.y = (resig, sr)
        self.ynumpy = self.y[0][0].numpy()
        return self.y

    def resample(self, newsr):
        sig, sr = self.y
        if (sr == newsr):
            return self.y
        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        self.y = (resig, newsr)
        self.ynumpy = self.y[0][0].numpy()
        self.sr = newsr
        return ((resig, newsr))
    
    def pad_trunc(self, max_ms):
        sig, sr = self.y
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms
        if (sig_len > max_len):
            sig = sig[:,:max_len]
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)

        self.y = (sig, sr)
        self.ynumpy = self.y[0][0].numpy()
        return self.y
    
    def time_shift(self, shift_limit):
        sig,sr = self.y
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)

        self.y = (sig.roll(shift_amt), sr)
        self.ynumpy = self.y[0][0].numpy()
        return self.y

    def spectro_gram(self, n_mels=64, n_fft=1024, hop_len=None, condensed = False):
        sig,sr = self.y
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        if condensed:
            spec = np.array(spec[0]).mean(axis=0)

        return np.array(spec)
    
    def MFCC_extraction(self, y_trim = 0, n_mfcc = 40, mean = True, remix = True, align_zero = True, max_ms = 5000):
        if self.ynumpy != None:
            y_trim = self.ynumpy
        
        if remix:
            y_trim = librosa.effects.remix(self.ynumpy, intervals=librosa.effects.split(self.ynumpy), align_zeros=align_zero)
            y_trim = self.pad_trunc_function(y_trim, self.sr, max_ms, 1)

        if mean:
            mfcc = np.mean(librosa.feature.mfcc(y=y_trim, sr=self.sr, n_mfcc= n_mfcc).T, axis = 0)
        else:
            mfcc = librosa.feature.mfcc(y=y_trim, sr=self.sr, n_mfcc= n_mfcc).T
        return mfcc

    def MFCC_delta(self, order = 1):
        delta = librosa.feature.delta(self.mfcc, order = order)
        return delta

    def spectro_augment(self, max_mask_pct=0.1,n_freq_masks=1,n_time_masks=1, condensed = False):
        spec = self.spectro_gram()
        novalue, n_mels, n_steps = np.array(spec).shape
        mask_value = spec.mean()
        aug_spec = torch.from_numpy(spec)

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        if condensed:
            aug_spec = np.array(aug_spec[0]).mean(axis=0)

        return np.array(aug_spec)
    

    def chroma_extraction(self):

        stft = np.abs(librosa.stft(self.ynumpy))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=self.sr).T,axis=0)
        return chroma

    def spectralContract_extraction(self):
        stft = np.abs(librosa.stft(self.ynumpy))
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=self.sr).T,axis=0)
        return contrast
    
    def tonalCentroid_extraction(self):
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(self.ynumpy),sr=self.sr).T,axis=0)
        return tonnetz

    def extract_all_features(self, is_concat = False, spectro_augment = False):
        mfcc = self.MFCC_extraction()
        chroma = self.chroma_extraction()
        if (spectro_augment):
            mel = self.spectro_augment(condensed=True)
        else:
            mel = self.spectro_gram(condensed=True)
        contrast = self.spectralContract_extraction()
        tonnetz = self.tonalCentroid_extraction()

        #mel = self.spectro_gram()
        #mel = np.array(self.spectro_augment(mel, n_freq_masks=2, n_time_masks=2))

        if is_concat == True:
            return np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
            
        return mfcc, chroma, mel, contrast, tonnetz

    def plot_spectrogram(self, augment = False):
        if (augment):
            librosa.display.specshow(self.spectro_augment()[0], sr=self.sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
        else:
            librosa.display.specshow(self.spectro_gram()[0], sr=self.sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')

    

    # Author's Notes: Seems like Pytorch does the melspectrogram better

    def melScaled_extraction(self):
        mel = np.mean(librosa.feature.melspectrogram(y = self.ynumpy, sr=self.sr, n_fft = 1024, n_mels= 64, hop_length=None), axis = 0) #np.mean(...(,axis = 0))
        return mel
    
    def pad_trunc_function(self, sig, sr, max_ms, num_rows = 1):
        sig_len = sig.shape[0]
        max_len = sr//1000 * max_ms
        if (sig_len > max_len):
            sig = sig[:,:max_len]
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = np.zeros((num_rows, pad_begin_len))
            pad_end = np.zeros((num_rows, pad_end_len))
            sig = np.concatenate((pad_begin[0], sig, pad_end[0]), 0)

        return sig
    