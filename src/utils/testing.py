import torch
from utils.clustering import ClusterModule
from utils.embedding import EmbedderModule
from utils.vad import VADModule
from utils.generalClasses import DataLoader_extraction
from utils.audioSplitter import SplitterModule

class TesterModule():
    def __init__(self) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, audio, window_length, vad, embedder, clusterer):

        window_size = int(window_length * 16000)

        #Get tensors from the audio path
        DL = DataLoader_extraction(audio)
        tensors = DL.y[0][0]

        #Voice Activation Detection (Modularise VAD class soon)
        VAD = VADModule(vad)
        vad_check, sampling_rate, _ = VAD.silero_vad_inference(tensors, window_size_samples= window_size)

        #Split the tensors into desired window_size
        Splitter = SplitterModule()
        split_tensors = Splitter.split_audio_tensor(audio, window_size/sampling_rate)

        #Get embeddings
        Embedder = EmbedderModule(embedder)
        embedding_list, index_list = self.get_embeddings_with_vad_check(split_tensors, vad_check, Embedder)

        #Cluster the embeddings 
        Clusterer = ClusterModule(embedding_list, clusterer, 2)
        combine_list = self.get_list_with_index_and_labels(index_list, Clusterer)

        #Create the final string for presentation
        final_string, final_list = self.get_final_string(combine_list, window_size/sampling_rate)

        return final_string
    
    def get_embeddings_with_vad_check(self, tensors, vad_check, embedder):

        features_list = []
        index_list = []

        for x in range(len(tensors)-1):
            if vad_check[x]:
                features = embedder.get_embeddings(tensors[x])
                features = features.cpu().detach()
                features = features.tolist()
                features_list.append(features[0][0])
                index_list.append(x)
        
        return features_list, index_list
    
    def get_list_with_index_and_labels(self, index_list, CM):
        combine_list = []

        for x in range(len(index_list)):
            combine_list.append([index_list[x], CM.get_labels()[x]])

        return combine_list
    
    def get_final_string(self, combine_list, length_of_interval):
        starting = 1
        final_string = ''
        final_list = []

        for x in range(len(combine_list)):
            if starting:
                start = combine_list[x][0] * length_of_interval
                final_string += f'Speaker {combine_list[x][1]} : {start}'
                starting = 0
            
            if x != len(combine_list)-1:
                if combine_list[x+1][1] != combine_list[x][1] or combine_list[x+1][0] != combine_list[x][0]+1:
                    end = (combine_list[x][0] + 1) * length_of_interval
                    final_string += f' - {end} \n'
                    starting = 1
                    final_list.append([combine_list[x][1], start, end])

            else:
                end = (combine_list[x][0] + 1) * length_of_interval
                final_string += f' - {end} \n'
                final_list.append([combine_list[x][1], start, end])

        return final_string, final_list
        