import simpleder
import os
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from constants.CALLHOME import CALLHOME_audio
from logzero import logger

class ScoringModule():

    def __init__(self, choice = 'Pyannote DER') -> None:

        self.name = None
        match choice:
            case 'Pyannote DER':
                self.name = choice


    def score(self, audio_path, testing):

        logger.info(audio_path)

        ground_truth_path = self.get_ground_truth_path(audio_path)

        ground_truth = self.get_ground_truth(ground_truth_path)

        if ground_truth == None:
            return

        hypothesis = Annotation()
        reference = Annotation()

        for x in testing:
            hypothesis[Segment(float(x[1]), float(x[2]))] = x[0]

        for x in ground_truth:
            reference[Segment(float(x[1]), float(x[2]))] = x[0]

        start = ground_truth[0][1]
        end = ground_truth[-1][2]
        diarizationErrorRate = DiarizationErrorRate(skip_overlap=True)

        return diarizationErrorRate(reference, hypothesis, uem=Segment(start, end), detailed=True)['diarization error rate']
    
    def get_ground_truth(self, path):

        match path[-3:]:

            case 'txt':

                return self.read_txt_to_list(path)

            case 'cha':

                return self.read_cha_to_list_CALLHOME(path)
            
        return None

    
    def get_ground_truth_path(self, audio_path):

        final_path = 'data/answer/'

        if 'british_ministers' in audio_path:
            return final_path + 'british_ministers.txt'
        
        else:
            for x in CALLHOME_audio:
                
                if x in audio_path:
                    return final_path+'CALLHOME/'+x+'.cha'
            
            return None
    
    def stringify(self, error_rate):

        return self.name + ' error rate : ' + str(error_rate) + '\n\n'


    def read_cha_to_list_CALLHOME(self, path_to_transcript):

        with open(path_to_transcript) as f:
            lines = f.readlines()

        temp_list = []
        cont = 0
        for x in range(len(lines)):
            if lines[x][0:2] == '*A' or lines[x][0:2] == '*B' or cont == 1:

                index_start = lines[x].find('\x15') + 1
                if cont == 0:
                    temp =lines[x][0:2] + ' '

                if index_start != 0:
                    for u in lines[x][index_start:]:
                        if u == '\x15':
                            break
                        temp+=u

                    temp_list.append(temp)
                
                    cont = 0

                else:
                    cont = 1
        
        ground_truth = []

        for r in temp_list:

            new_list = []

            start_end = r.find(' ')
            underscore_end = r.find('_')
            true_end = len(r)

            new_list = (r[1], float(r[start_end+1:underscore_end-3]+'.'+r[underscore_end-4:underscore_end:6]), 
                        float(r[underscore_end+1:true_end-3]+'.'+r[underscore_end-2:underscore_end]))

            ground_truth.append(new_list)

        if len(ground_truth) == 0:
            os.remove(path_to_transcript)
            return None

        return ground_truth
    
    def read_txt_to_list(self, path):
    
        with open(path) as f:

            read = f.readlines()

            final_list = []

            for x in read:
                final_list.append(eval(x))

            return final_list
