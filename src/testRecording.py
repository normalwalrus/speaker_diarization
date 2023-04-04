from utils.audioRecorder import RecorderModule
from logzero import logger

if __name__ == "__main__":

    logger.info('Recording starting...')

    Recorder = RecorderModule()
    length = 10
    name = 'test'
    sampling_rate = 16000
    channel = 1

    Recorder.record_audio(length, name, sampling_rate, channel)

    logger.info('Recording done and saved...')

