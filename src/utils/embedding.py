import torch
from speechbrain.pretrained.interfaces import Pretrained

class EmbedderModule():
    def __init__(self, choice = 'ECAPA_TDNN_pretrained') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = choice
        
        match choice:

            case 'ECAPA_TDNN_pretrained':
                self.classifier = Encoder_ECAPA_TDNN.from_hparams(
                    source="yangwang825/ecapa-tdnn-vox2"
                ).to(self.device)
                self.choice = choice

            case _:
                print('Error: Embedder Choice not found')

    def get_embeddings(self, tensors):

        features = torch.tensor(self.classifier.encode_batch(tensors[0], device = self.device))

        return features


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