from utils.generalClasses import DataLoader_extraction, Encoder
import torch

class EmbedderModule():
    def __init__(self, choice = 'ECAPA_TDNN_pretrained') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        match choice:

            case 'ECAPA_TDNN_pretrained':
                self.classifier = Encoder.from_hparams(
                    source="yangwang825/ecapa-tdnn-vox2"
                ).to(self.device)
                self.choice = choice

            case _:
                print('Error: Embedder Choice not found')

    def get_embeddings(self, tensors):

        features = torch.tensor(self.classifier.encode_batch(tensors[0], device = self.device))

        return features
