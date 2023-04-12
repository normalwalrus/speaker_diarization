import simpleder

class ScoringModule():

    def __init__(self, choice = 'simpleder') -> None:

        self.name = None
        match choice:
            case 'simpleder':
                self.name = choice


    def score(self, audio_path, testing):

        ground_truth_path = self.get_ground_truth_path(audio_path)

        ground_truth = self.get_ground_truth(ground_truth_path)

        return simpleder.DER(ground_truth, testing)
    
    def get_ground_truth(self, path):

        with open(path) as f:
            read = f.readlines()

        final_list = []

        for x in read:
            final_list.append(eval(x))

        return final_list
    
    def get_ground_truth_path(self, audio_path):

        final_path = 'data/answer/'

        if 'british_ministers' in audio_path:
            return final_path + 'british_ministers.txt'
            
        return None
    
    def stringify(self, error_rate):

        return self.name + ' error rate : ' + str(error_rate) + '\n\n'


