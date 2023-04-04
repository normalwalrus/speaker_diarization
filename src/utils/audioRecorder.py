import sounddevice as sd
import os
from scipy.io.wavfile import write

class RecorderModule():

    def __init__(self) -> None:
        pass

    def record_audio(self, length : int, name:str, sample_rate = 16000, channel = 1):
        
        final_dest_path = os.getcwd() + '/audio/'+name+'.wav'

        recording = sd.rec(int(length * sample_rate),samplerate=sample_rate, channels=channel)
        sd.wait()

        write(final_dest_path, sample_rate, recording)

        return