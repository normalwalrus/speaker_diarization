import numpy as np
import librosa
import torch
from logzero import logger

class SplitterModule:

    """
    Class is used to read audio from the path provided, split it into intervals (Default is 5 seconds)
    and produce a resultant array of Tensors
    """

    def split_audio_tensor(self, path, segment_dur_secs = 5):
        """
        Function used to read audio from the path provided, split it into intervals (Default is 5 seconds)
        and produce a resultant array of Tensors

        Parameters
        ----------
            path : String
                Path to the audio file

            segment_dur_secs : Integer (Default = 5)
                Duration of resultant tensors from audio clip

        Returns
        ----------
            split: List
                List of Tensors from the Audio Clip
        """

        y = librosa.load(path, sr = None, mono = True)

        segment_length = int(y[1] * segment_dur_secs)
        num_sections = int(np.ceil(len(y[0]) / segment_length))
        
        split = []

        for i in range(num_sections):
            t = y[0][i * segment_length: (i + 1) * segment_length]
            split.append([torch.tensor([t]), y[1]])
            
        return split