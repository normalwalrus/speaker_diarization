import simpleder
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

class ScoringModule():

    def __init__(self, choice = 'Pyannote DER') -> None:

        self.name = None
        match choice:
            case 'Pyannote DER':
                self.name = choice


    def score(self, audio_path, testing):

        ground_truth_path = self.get_ground_truth_path(audio_path)

        ground_truth = self.get_ground_truth(ground_truth_path)

        hypothesis = Annotation()
        reference = Annotation()

        for x in testing:
            hypothesis[Segment(float(x[1]), float(x[2]))] = x[0]

        for x in ground_truth:
            reference[Segment(float(x[1]), float(x[2]))] = x[0]

        start = ground_truth[0][1]
        end = ground_truth[-1][2]
        diarizationErrorRate = DiarizationErrorRate()

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
        
        elif '0638' in audio_path:
            return final_path+'CALLHOME/0638.cha'
        
        elif '4074' in audio_path:
            return final_path+'CALLHOME/4074.cha'
            
        return None
    
    def stringify(self, error_rate):

        return self.name + ' error rate : ' + str(error_rate) + '\n\n'


    def read_cha_to_list_CALLHOME(self, path_to_transcript):

        with open(path_to_transcript) as f:
            lines = f.readlines()

        temp_list = []
        for x in lines:
            if x[0:2] == '*A' or x[0:2] == '*B':

                index_start = x.find('\x15') + 1

                if index_start != 0:
                    temp =x[0:2] + ' '
                    for u in x[index_start:]:
                        if u == '\x15':
                            break
                        temp+=u

                    temp_list.append(temp)

        ground_truth = []

        for r in temp_list:

            new_list = []

            start_end = r.find(' ')
            underscore_end = r.find('_')
            true_end = len(r)-3

            new_list = (r[1], float(r[start_end+1:underscore_end-3]), float(r[underscore_end+1:true_end]))

            ground_truth.append(new_list)

        return ground_truth
    
    def read_txt_to_list(self, path):
    
        with open(path) as f:

            read = f.readlines()

            final_list = []

            for x in read:
                final_list.append(eval(x))

            return final_list
