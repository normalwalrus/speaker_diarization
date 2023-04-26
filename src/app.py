import gradio as gr
from utils.testing import TesterModule
import constants.messages as messages

path_to_audio = 'data/audio/'

vad_choices = ['silero-vad']
embedder_choices = ['ECAPA_TDNN_pretrained', 'ECAPA_TDNN_pretrained_singaporean', 'Wav2Vec2', 'MFCC', 'titanet-l.nemo']
clustering_choices = ['KMeans', 'Spectral', 'Agglomerative', 'Google_Spectral']
dataset_choices = ['None', 'CALLHOME', 'Noised_CALLHOME', 'Chatter_CALLHOME']

EXAMPLES = [[path_to_audio+'british_ministers.wav', 2, 1, 'silero-vad', 'titanet-l.nemo', 'Google_Spectral', False, True, 'CALLHOME'],
            [path_to_audio+'british_ministers.wav', 2, 1, 'silero-vad', 'ECAPA_TDNN_pretrained', 'KMeans', False],
            [path_to_audio+'CALLHOME/0638.wav', 2, 1, 'silero-vad', 'ECAPA_TDNN_pretrained', 'KMeans', False],
            [path_to_audio+'CALLHOME/4074.wav', 2, 1, 'silero-vad', 'ECAPA_TDNN_pretrained', 'KMeans', False],
            [path_to_audio+'Sg_parliament.wav', 2, 1, 'silero-vad', 'ECAPA_TDNN_pretrained', 'Spectral', False]]


inputs = [gr.Audio(source='upload', type='filepath', label = 'Audio'),
          gr.Slider(0 ,5, step = 1, label = 'Number of Clusters (if 0 = for model inference)'),
          gr.Slider(0.5, 2, step = 0.1, label= 'Window Length (Sec)'),
          gr.Radio(vad_choices, label= 'VAD choice'), gr.Radio(embedder_choices, label='Embedder Choice'), 
          gr.Radio(clustering_choices, label='Clustering Choice'),
          gr.Checkbox(label = 'Transcription'), gr.Checkbox(label= 'DER check CALLHOME dataset'), 
          gr.Radio(dataset_choices, label='Clustering Choice')]

outputs = ['text']

if __name__ == "__main__":

    Tester = TesterModule()

    app = gr.Interface(
        Tester.main,
        inputs=inputs,
        outputs=outputs,
        title=messages.TITLE,
        description=messages.NULL,
        examples= EXAMPLES
    ).launch(server_name="0.0.0.0")
    