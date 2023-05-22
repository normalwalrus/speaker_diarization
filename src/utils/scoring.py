import simpleder
import os
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from constants.CALLHOME import CALLHOME_audio
from logzero import logger

class ScoringModule():
    """
    Class is used to select the scoring method to use. Be it Voice Activity Detection (VAD) or Diarization Error Rate (DER)
    """
    def __init__(self, choice = 'Pyannote DER') -> None:
        

        self.name = None
        match choice:
            case 'Pyannote DER':
                self.name = choice


    def score(self, audio_path, testing, assessment):
        """
        Choose which testing method to choose from 2 error rates

        Parameters
        ----------
        audio_path : String
            Path of the audio that is to be scored
        testing : List
            List of diarization ground truths (format : '[('A', 0.0, 3.0)]')
        assessment : String
            'DER' or 'VAD' currently to choose the assessment method
            
        Returns
        ----------
        error : float
            error rate value for chosen assessment method
        """
        match assessment:

            case 'DER':

                return self.DER_test(audio_path, testing)

            case 'VAD':

                return self.VAD_test(audio_path, testing)
        
    def VAD_test(self, audio_path, testing):
        """
        For Voice Activity Detection testing 

        Parameters
        ----------
        audio_path : String
            Path of the audio that is to be scored
        testing : List
            List of diarization ground truths (format : '[('A', 0.0, 3.0)]')
            
        Returns
        ----------
        error : float
            error rate value for chosen assessment method
        """

        logger.info(audio_path)

        ground_truth_path = self.get_ground_truth_path(audio_path)

        ground_truth = self.get_ground_truth(ground_truth_path)

        if ground_truth == None:
            return

        hypothesis = Annotation()
        reference = Annotation()

        for x in testing:
            hypothesis[Segment(float(x[1]), float(x[2]))] = 'A'

        for x in ground_truth:
            reference[Segment(float(x[1]), float(x[2]))] = 'A'

        start = ground_truth[0][1]
        end = ground_truth[-1][2]
        diarizationErrorRate = DiarizationErrorRate(skip_overlap=True)

        return diarizationErrorRate(reference, hypothesis, uem=Segment(start, end), detailed=True)['diarization error rate']
    
    def DER_test(self, audio_path, testing):
        """
        For Diarization Error Rate testing 

        Parameters
        ----------
        audio_path : String
            Path of the audio that is to be scored
        testing : List
            List of diarization ground truths (format : '[('A', 0.0, 3.0)]')
            
        Returns
        ----------
        error : float
            error rate value for chosen assessment method
        """

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
        """
        For formatting .cha file or .txt file to the appropriate python list for error rate test

        Parameters
        ----------
        path : String
            Path of the audio that is to be scored
            
        Returns
        ----------
        : list
            List of properly formatted list for scoring (format : '[('A', 0.0, 3.0)]')
            or None
        """

        match path[-3:]:

            case 'txt':

                return self.read_txt_to_list(path)

            case 'cha':

                return self.read_cha_to_list_CALLHOME(path)
            
        return None

    
    def get_ground_truth_path(self, audio_path):
        """
        To get the .cha or .txt file for the ground truth in scoring 
        
        Parameters
        ----------
        audio_path : String
            Path of the audio that is to be scored
            
        Returns
        ----------
        : string
            Path to the correct ground_truth .cha or .txt
        """
        final_path = 'data/answer/'

        if 'british_ministers' in audio_path:
            return final_path + 'british_ministers.txt'
        
        else:
            for x in CALLHOME_audio:
                
                if x in audio_path:
                    return final_path+'CALLHOME/'+x+'.cha'
            
            return None
    
    def stringify(self, error_rate):
        """
        Use the error rate to make a string for display
        
        Parameters
        ----------
        error rate : float
            value of the error rate from the scoring module
            
        Returns
        ----------
        : string
            Error rate string 
        """

        return self.name + ' error rate : ' + str(error_rate) + '\n\n'


    def read_cha_to_list_CALLHOME(self, path_to_transcript):
        """
        Convert the .cha file from CALLHOME labelled dataset to list for testing
        
        Parameters
        ----------
        path_to_transcript : string
            path to the .cha file with the correct transcript and timings for assessment
            
        Returns
        ----------
        ground_truth : list
            list of properly formatted audio segments to use as ground truth in the assessment
        """

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

            
            new_list = (r[1], float(r[start_end+1:underscore_end-3]+'.'+r[underscore_end-3:underscore_end]), 
                                float(r[underscore_end+1:true_end-3]+'.'+r[true_end-3:true_end]))

            ground_truth.append(new_list)

        if len(ground_truth) == 0:
            os.remove(path_to_transcript)
            return None

        return ground_truth
    
    def read_txt_to_list(self, path):
        """
        Convert the .txt file to list for testing
        
        Parameters
        ----------
        path : string
            path to the .txt file with the correct transcript and timings for assessment
            
        Returns
        ----------
        final_list : list
            list of properly formatted audio segments to use as ground truth in the assessment
        """
    
        with open(path) as f:

            read = f.readlines()

            final_list = []

            for x in read:
                final_list.append(eval(x))

            return final_list
