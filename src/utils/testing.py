import torch
import os
from utils.clustering import ClusterModule
from utils.embedding import EmbedderModule
from utils.vad import VADModule
from utils.audioDataloader import DataLoader_extraction
from utils.audioSplitter import SplitterModule
from utils.transcription import TransciptionModule
from utils.scoring import ScoringModule
from logzero import logger

class TesterModule():
    def __init__(self) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, audio, n_clusters, window_length, vad, embedder, clusterer, transcription):

        window_size = int(window_length * 16000)

        #Get tensors from the audio path
        logger.info('Extracting Features in Tensor form...')
        DL = DataLoader_extraction(path = audio)
        tensors = DL.y[0][0]

        #Voice Activation Detection (Modularise VAD class soon)
        logger.info('Performing Voice Activity Detction...')
        VAD = VADModule(vad)
        vad_check, sampling_rate, _ = VAD.silero_vad_inference(tensors, window_size_samples= window_size)

        #Split the tensors into desired window_size
        logger.info(f'Splitting audio into {window_size/sampling_rate} intervals...')
        Splitter = SplitterModule()
        split_tensors = Splitter.split_audio_tensor(audio, window_size/sampling_rate)

        #Get embeddings
        logger.info(f'Getting embeddings from {embedder}...')
        Embedder = EmbedderModule(embedder)
        embedding_list, index_list = self.get_embeddings_with_vad_check(split_tensors, vad_check, Embedder)

        #Cluster the embeddings 
        logger.info(f'Clusering using {clusterer}...')
        Clusterer = ClusterModule(embedding_list, clusterer, n_clusters)
        labels = Clusterer.get_labels()
        combine_list = self.get_list_with_index_and_labels(index_list, labels)

        #Create the final string for presentation
        logger.info(f'Forming final list for display...')
        final_string, final_list, for_assessing = self.get_final_string_without_transcription(combine_list, window_size/sampling_rate)

        #Assessing error rate of the resultant list of tuples
        scorer = ScoringModule()
        if scorer.get_ground_truth_path(audio):
            error_rate = scorer.score(audio, for_assessing)
            final_string = scorer.stringify(error_rate) + final_string

        if not (transcription):
            logger.info(f'Display final string without transciption...')
            return final_string

        #Adding Transcriptions
        logger.info(f'Transcripting audio...')
        Transcriber = TransciptionModule()
        transcribed_list = Transcriber.diarization_transcription(final_list, split_tensors, [window_size, sampling_rate])

        #Form final list
        logger.info(f'Display final string with transciption...')
        transcribed_string = self.get_final_string_with_transcription(transcribed_list, final_list)

        logger.info('Exiting out of predict...')
        return transcribed_string
    
    def export_textfile(self, list_temp, name, path = '/../data/text/'):
        with open(os.getcwd()+f'/data/text/{name}.txt', 'w') as f:
            for line in list_temp:
                f.write(f"{line}\n")
    
    def get_final_string_with_transcription(self, transcibed_list, final_list):

        final_string = ''
        for x in range(len(final_list)):
            final_list[x].append(transcibed_list[x])

        for x in range(len(final_list)):
            
            temp = f'Speaker {final_list[x][0]} : {final_list[x][1]}s - {final_list[x][2]}s \n {final_list[x][3]} \n\n'
            final_string+=temp

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
    
    def get_list_with_index_and_labels(self, index_list, labels):
        combine_list = []

        for x in range(len(index_list)):
            combine_list.append([index_list[x], 'A' if labels[x]==0 else 'B'])

        return combine_list
    
    def get_final_string_without_transcription(self, combine_list, length_of_interval):
        starting = 1
        final_string = ''
        final_list = []
        for_assessing = []

        for x in range(len(combine_list)):
            if starting:
                start = combine_list[x][0] * length_of_interval
                final_string += f'Speaker {combine_list[x][1]} : {round(start,2)}'
                starting = 0
            
            if x != len(combine_list)-1:
                if combine_list[x+1][1] != combine_list[x][1] or combine_list[x+1][0] != combine_list[x][0]+1:
                    end = (combine_list[x][0] + 1) * length_of_interval
                    final_string += f' - {round(end,2)} \n'
                    starting = 1
                    final_list.append([combine_list[x][1], round(start,2), round(end,2)])
                    for_assessing.append((combine_list[x][1], round(start,2), round(end,2)))

            else:
                end = (combine_list[x][0] + 1) * length_of_interval
                final_string += f' - {round(end,2)} \n'
                final_list.append([combine_list[x][1], round(start,2), round(end,2)])
                for_assessing.append((combine_list[x][1], round(start,2), round(end,2)))

        return final_string, final_list, for_assessing