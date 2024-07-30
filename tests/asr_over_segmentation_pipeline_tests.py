import pickle
from pathlib import Path

import ctranslate2
import pytest
from pyannote.core.annotation import Annotation
from pyannote.audio import Audio as pyannAudio
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

from whispertd.utils.utils import download_model
from whispertd.pipelines.asr_over_segmentation import ASRoverSegmentationCt2Pipeline
from whispertd.models.modeling_ct2_utils import Ct2PreTrainedModel

WHISPER_MODEL_VERSION = 'tiny'


"""
1. **audio_data**: Фикстура для загрузки и нормализации аудио данных.
2. **diarization**: Фикстура для загрузки данных диаризации.
3. **model_ct2**: Фикстура для инициализации модели.
4. **tokenizer**: Фикстура для инициализации токенизатора.
5. **feature_extractor**: Фикстура для инициализации извлекателя признаков.
6. **pipeline**: Фикстура для инициализации пайплайна с использованием вышеописанных компонентов.
7. **test_init_ct2pipeline**: Тест для проверки инициализации пайплайна.
8. **test_AsrOverSegmentationPipeline**: Параметризованный тест для проверки работы пайплайна с разными значениями `batch_size`.
"""
@pytest.fixture
def audio_data():
    input_dir = './data'
    in_file_m4a = f'{input_dir}/20230316_203920 Мише скучно в садике_mono.wav'
    audio_loader = pyannAudio(sample_rate=16000, backend='soundfile')
    waveform_file_m4a, sample_rate_file_m4a = audio_loader(in_file_m4a)
    waveform_file_m4a = audio_loader.power_normalize(waveform_file_m4a)
    return waveform_file_m4a, in_file_m4a


@pytest.fixture
def diarization(audio_data):
    input_dir = './data'
    in_file_m4a = audio_data[1]
    out_file_m4a = Path(input_dir).joinpath(Path(in_file_m4a).name)
    with open(f'{out_file_m4a}_dia.pickle', 'rb') as file:
        diarization: Annotation = pickle.load(file)
    return diarization


@pytest.fixture
def model_ct2():
    model_path = download_model(f'{WHISPER_MODEL_VERSION}')
    return ctranslate2.models.Whisper(model_path, device='cpu')


@pytest.fixture
def tokenizer():
    return WhisperTokenizerFast.from_pretrained(f"openai/whisper-{WHISPER_MODEL_VERSION}",
                                                language="ru",
                                                task='transcribe')


@pytest.fixture
def feature_extractor():
    return WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{WHISPER_MODEL_VERSION}")


@pytest.fixture
def pipeline(model_ct2, tokenizer, feature_extractor):
    return ASRoverSegmentationCt2Pipeline(model_ct2, tokenizer, feature_extractor)


def test_init_ct2pipeline():
    """
    Test initializing of base class Pipeline and specific Ct2Pipeline arguments

    :return:
    """
    model_ct2 = Ct2PreTrainedModel()
    ct2pipeline = ASRoverSegmentationCt2Pipeline(model=model_ct2)
    assert isinstance(ct2pipeline, ASRoverSegmentationCt2Pipeline)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 7])
def test_AsrOverSegmentationPipeline(batch_size, audio_data, diarization, pipeline):
    waveform_file_m4a = audio_data[0]
    result = []
    for out in pipeline(waveform_file_m4a[0].numpy()[:128000],
                       segmentation=diarization,
                       min_duration_on=0.250,
                       min_duration_off=2.0,
                       batch_size=batch_size
                       ):
        result.append(out)
        print(out)
    assert len(result) > 0


if __name__ == '__main__':
    pytest.main()
