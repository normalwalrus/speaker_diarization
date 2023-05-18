import sounddevice as sd
import os
from scipy.io.wavfile import write

class RecorderModule():

    """
    Class is used to record audio
    """

    def __init__(self) -> None:
        pass

    def record_audio(self, length : int, name:str, sample_rate = 16000, channel = 1):
        """
        Function used to record audio

        Parameters
        ----------
            length : Integar
                Length of the audio in seconds
            name : String
                Name of the audio clip that will be produced
            sample_rate : Integar
                Sample rate of the audio
            channel : Intehar
                2 being Stereo, 1 being mono
        """
        
        final_dest_path = os.getcwd() + '/audio/'+name+'.wav'

        recording = sd.rec(int(length * sample_rate),samplerate=sample_rate, channels=channel)
        sd.wait()

        write(final_dest_path, sample_rate, recording)

        return