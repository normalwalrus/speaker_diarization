import torch
import os
import librosa
import numpy as np
import torchaudio
import nemo.collections.asr as nemo_asr
from utils.audioDataloader import audio_dataloader
from speechbrain.pretrained.interfaces import Pretrained
from models.ECAPA_TDNN import ECAPA_TDNN
from models.neuralnet import FeedForwardNN

PATH_TO_ECAPA_TDNN = '/models/ECAPA_TDNN_v1.0_5sec_80MFCC_30epoch.pt'
PATH_TO_FEEDFORWARD = os.getcwd()+'/models/ECAPA_TDNN_Pretrained_v1.0_5sec_10epoch.pt'

class EmbedderModule():
    """
    Class is used to select the embedding choice and get the embedding values
    """
    def __init__(self, choice = 'ECAPA_TDNN_pretrained') -> None:
        """
        Initialises the embedding module

        Parameters
        ----------
            choice: String
                Choice of the embedding methods that are stated in the module
                Choose from ['ECAPA_TDNN_pretrained_singaporean','ECAPA_TDNN_pretrained', 'Wav2Vec2', 'titanet-l.nemo', 'MFCC' ]
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = choice
        
        match choice:

            case 'ECAPA_TDNN_pretrained_singaporean':
                self.classifier = Encoder_ECAPA_TDNN.from_hparams(
                    source="yangwang825/ecapa-tdnn-vox2"
                ).to(self.device)
                self.embedder = FeedForwardNN(192, 20, 0).to(self.device)
                self.embedder.load_state_dict(torch.load(PATH_TO_FEEDFORWARD))
                self.embedder.eval().double()

            case 'ECAPA_TDNN_pretrained':
                self.classifier = Encoder_ECAPA_TDNN.from_hparams(
                    source="yangwang825/ecapa-tdnn-vox2"
                ).to(self.device)

            case 'Wav2Vec2':
                bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
                self.classifier = bundle.get_model()

            case 'titanet-l.nemo':
                self.classifier = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path = os.getcwd() + "/models/titanet-l.nemo")
            
            case 'MFCC':
                self.classifier = None

            case _:
                print('Error: Embedder Choice not found')

    def get_embeddings(self, tensors):
        """
        Put the tensors (function built to receive tensors from audio wave) through the embedding module to get embeddings

        Parameters
        ----------
            tensors: Torch.tensor
                Tensors (function built to receive tensors from audio wave) through the embedding module to get embeddings
        Returns
        ----------
            features : Torch.tensor
                Torch.tensor with the embeddings from the chosen embedder
        """

        match self.name:

            case 'ECAPA_TDNN_pretrained_singaporean':
                features = torch.tensor(self.classifier.encode_batch(tensors[0], device = self.device))
                features = features.type(torch.double).to(self.device)
                features = torch.tensor(self.embedder(features))

            case 'ECAPA_TDNN_pretrained':
                features = torch.tensor(self.classifier.encode_batch(tensors[0], device = self.device))
            
            case 'titanet-l.nemo':
                torchaudio.save('test.wav', tensors[0], 16000)
                features = self.classifier.get_embedding('test.wav')
                features = features[None, :]
                os.remove('test.wav')

            case 'Wav2Vec2':
                features = self.classifier.extract_features(tensors[0].type(torch.float))
                final = np.array([])

                for x in features[0]:
                    x = x[0].detach().numpy()

                    x = np.mean(x, axis=1)

                    final = np.concatenate((final, x), axis = None)

                features = torch.from_numpy(final)
                features = features[None, None, :]
            
            case 'MFCC':

                DL = audio_dataloader(sr=16000)
                features = tensors[0].numpy()
                features = DL.MFCC_extraction(features, mean = False, remix = False)
                features = torch.from_numpy(features)

            case 'ECAPA_TDNN': # DOES NOT WORK
                tensors = tensors[0].numpy()
                features = self.get_MFCC(tensors)
                print(features.shape)
                features = self.classifier([features])

        return features
    
    def get_MFCC(self, audio_np, n_mfcc = 80, mean = False, sr = 16000):

        if mean:
            mfcc = np.mean(librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc= n_mfcc).T, axis = 0)
        else:
            mfcc = librosa.feature.mfcc(y = audio_np, sr=sr, n_mfcc= n_mfcc).T
        return mfcc


class Encoder_ECAPA_TDNN(Pretrained):
    """
    Class used to instantiate the pretrained ECAPA_TDNN
    """

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False, device = 'cpu'):
        """
        Get the speaker embeddings from the ECAPA_TDNN

        Parameters
        ----------
            tensors: Torch.tensor
                Tensors (function built to receive tensors from audio wave) through the embedding module to get embeddings
        Returns
        ----------
            features : Torch.tensor
                Torch.tensor with the embeddings from the chosen embedder
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(device), wav_lens.to(device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings,
                torch.ones(embeddings.shape[0], device=device)
            )
        return embeddings