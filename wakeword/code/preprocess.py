import torchaudio
import numpy as np


class Loader():
    """ Loader loads audio data, resamples if neccessary """

    def __init__(self, sample_rate, duration):
        self.sample_rate = sample_rate
        self.duration = duration

    def load(self, file_path):
        waveform, sr = torchaudio.load(file_path, normilization=False)
        if sr > self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform


class Padder():
    """ Padder pads loaded data, split between left and right pad """

    def __init__(self, mode="constant", npad):
        self.mode = mode
        self.npad = npad

    def right_pad(self, array, npad):
        padded_array = np.pad(array, (0, npad), mode = self.mode)
        return padded_array
    
    def left_pad(self, array, npad):
        padded_array = np.pad(array, (npad, 0), mode = self.mode)
        return padded_array


class SpecExtractor():
    """ SpecExtractor extracts spectrogram data from wavedata """

    def __init__(self, nfft, hop_length):
        self.nfft = nfft
        self.hop_length = hop_length
    
    def extract(self, signal):
        stft = torch.stft(signal, n_fft=self.nfft, hop_length=self.hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        return log_spectrogram


class MinMaxNorm():
    """ MinMaxNorm normalizes the array using min-max-norm """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, array):
        mn, mx = array.min(), array.max()
        zo_norm = (array - mn) / (mx - mn)                                  # norms array [0,1]
        norm_array = zo_norm * (self.max_val - self.min_val) + self.min_val # norms array [min,max]
        return norm_array
    
    def denormalize(self, norm_array, orig_min, orig_max):
        mn, mx = array.min(), array.max()
        zo_norm = (norm_array - self.min) / (self.max_val - self.min_val)
        array = zo_norm * (orig_max - orig_min) + orig_min
        return array 



    
