import os
import math
import torch
import torchaudio
import pytorch_lightning as pl
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils
from dotenv import load_dotenv

class TransciptionModule:

    """
    This class loads in the conformer CTC model from the model folder. Model is used for 
    preprocessing and postprocessing of the audio signal that is supplied. 

    """

    def __init__(self) -> None:

        self.model_path = os.getcwd() + '/models/stt_en_conformer_ctc_large.nemo'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.asr_model = self.initialize_asr_model()
    
    def initialize_asr_model(self) -> ASRModel:
        """
        Restores the stt_en_conformer_ctc into an ASRModel abstract class. This set up
        allows for us to use the preprocessing and postprocessing functions inbuilt.

        Since the model will always be this conformer model, there is no need for parameters

        Returns
        ----------
        ASRModel
            ASRModel that has the conformer model restored into it
        """
        model_cfg = ASRModel.restore_from(restore_path=self.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath) 
        asr_model = imported_class.restore_from(
            restore_path=self.model_path, map_location=self.device,
        )

        trainer = pl.Trainer(devices=1, accelerator=self.device)
        asr_model.set_trainer(trainer)
        asr_model = asr_model.eval()

        return asr_model
    
    def diarization_transcription(self, final_list, tensors, params = [16000, 16000]):
        transciption = []
        window, sample_rate = params
        basic_unit = window/sample_rate
        count = -1

        for x in final_list:
            temp_audio = torch.tensor([])
            period = x[2] - x[1]
            no_of_units = math.ceil(period/basic_unit)

            for y in range(no_of_units):
                temp_audio = torch.cat((temp_audio, tensors[count+y][0]),1)

            count+=no_of_units
            
            name = 'test.wav'
            torchaudio.save(name, temp_audio, 16000)
            transciption.append(self.asr_model.transcribe([name]))
            os.remove(name)

        return transciption