import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.generalClasses import DataLoader_extraction

class VADModule():

    def __init__(self, choice) -> None:

        match choice:
            case 'silero-vad':
                self.model, _ = torch.hub.load(repo_or_dir='models/silero-vad',
                                                    source = 'local',
                                                        model='silero_vad',
                                                        force_reload=True)

        self.sampling_rate = 16000

    def silero_vad_inference(self, tensor, window_size_samples = 8000, threshold = 0.2, plot = False):
  
        speech_probs = []
        sampling_rate = self.sampling_rate

        for i in range(0, len(tensor), window_size_samples):
                if len(tensor[i: i+ window_size_samples]) < window_size_samples:
                    break
                speech_prob = self.model(tensor[i: i+ window_size_samples], sampling_rate).item()
                speech_probs.append(speech_prob)

        self.model.reset_states()
        audio_length = len(speech_probs)/(sampling_rate/window_size_samples)

        aud_speech = []

        for x in speech_probs:
            if x > threshold:
                aud_speech.append(1)
            else:
                aud_speech.append(0)

        if plot:
            self.visualise(aud_speech, audio_length, sampling_rate)

        return aud_speech, sampling_rate, audio_length
    
    def visualise(self, aud_speech, audio_length, sampling_rate):
        # data to be plotted
        window_size_samples = (audio_length*sampling_rate)/len(aud_speech)
        num_samples = int(audio_length * sampling_rate)
        x = np.arange(num_samples/window_size_samples) / (sampling_rate/window_size_samples)
        y = np.array(aud_speech)

        # plotting
        plt.title("Line graph")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(x, y, color ="red")

        plt.show()

        return
