import pickle
from pathlib import Path

import ctranslate2
from pyannote.core.annotation import Annotation
from pyannote.audio import Audio as pyannAudio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperTokenizerFast

from whispertd.utils.utils import download_model

from whispertd.pipelines.asr_over_segmentation import ASRoverSegmentationCt2Pipeline
from whispertd.models.modeling_ct2_utils import Ct2PreTrainedModel

WHISPER_MODEL_VERSION = 'tiny'


def test_init_ct2pipeline():
    """
    Test initializing of base class Pipeline and specific Ct2Pipeline arguments

    :return:
    """
    model_ct2 = Ct2PreTrainedModel()
    ct2pipeline = ASRoverSegmentationCt2Pipeline(model=model_ct2)
    print(ct2pipeline)


def test_ASRoverSegmentationCt2Pipeline():
    input_dir = './data'
    in_file_m4a = f'{input_dir}/20230316_203920 Мише скучно в садике_mono.wav'

    audio_loader = pyannAudio(sample_rate=16000, backend='soundfile')

    waveform_file_m4a, sample_rate_file_m4a = audio_loader(in_file_m4a)
    waveform_file_m4a = audio_loader.power_normalize(waveform_file_m4a)

    out_file_m4a = Path(input_dir).joinpath(Path(in_file_m4a).name)
    with open(f'{out_file_m4a}_dia.pickle', 'rb') as file:
        diarization: Annotation = pickle.load(file)

    model_path = download_model(f'{WHISPER_MODEL_VERSION}')
    model_ct2 = ctranslate2.models.Whisper(model_path, device='cpu')

    tokenizer = WhisperTokenizerFast.from_pretrained(f"openai/whisper-{WHISPER_MODEL_VERSION}",
                                                     language="ru",  # language='russian')
                                                     task='transcribe',
                                                     )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{WHISPER_MODEL_VERSION}")

    pipe = ASRoverSegmentationCt2Pipeline(model_ct2, tokenizer, feature_extractor, batch_size=2)
    results = pipe(waveform_file_m4a[0].numpy()[:128000],
                   segmentation=diarization,
                   min_duration_on=0.250,
                   min_duration_off=2.0,
                   )
    print(results)


if __name__ == '__main__':
    #
    # test(diarization)

    # test_init_ct2pipeline()
    test_ASRoverSegmentationCt2Pipeline()
