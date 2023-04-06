import torch
import os
import librosa
import numpy as np
import torchaudio
from utils.audioDataloader import DataLoader_extraction
from speechbrain.pretrained.interfaces import Pretrained
from models.ECAPA_TDNN import ECAPA_TDNN

PATH_TO_ECAPA_TDNN = '/models/ECAPA_TDNN_v1.0_5sec_80MFCC_30epoch.pt'

class EmbedderModule():
    def __init__(self, choice = 'ECAPA_TDNN_pretrained') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = choice
        
        match choice:

            case 'ECAPA_TDNN_pretrained':
                self.classifier = Encoder_ECAPA_TDNN.from_hparams(
                    source="yangwang825/ecapa-tdnn-vox2"
                ).to(self.device)

            case 'Wav2Vec2':
                bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
                self.classifier = bundle.get_model()
            
            case 'MFCC':
                self.classifier = None
            
            case 'ECAPA_TDNN': # DOES NOT WORK SINCE ECAPA TRAINED ON 5 SEC AUDIO ONLY
                save_path = os.getcwd() + PATH_TO_ECAPA_TDNN
                self.classifier = ECAPA_TDNN(157 ,512, 20)
                self.classifier.load_state_dict(torch.load(save_path))
                self.classifier.eval().double()
                self.datatype = 'MFCC'

            case _:
                print('Error: Embedder Choice not found')

    def get_embeddings(self, tensors):

        match self.name:

            case 'ECAPA_TDNN_pretrained':
                features = torch.tensor(self.classifier.encode_batch(tensors[0], device = self.device))

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

                DL = DataLoader_extraction(sr=16000)
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

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False, device = 'cpu'):
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