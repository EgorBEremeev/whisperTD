# The MIT License (MIT)
#
# Copyright (c) 2024- Egor B Eremeev (egor.b.eremeev@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""ASR over Speaker diarization pipeline"""
import os
import types

import ctranslate2
from ctranslate2 import StorageView
from ctranslate2.models import WhisperGenerationResult
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Pipeline, PreTrainedTokenizer, WhisperTokenizer, WhisperFeatureExtractor, WhisperTokenizerFast

import torch
import gc

from pyannote.core import Segment
from pyannote.core.annotation import Annotation
from pyannote.audio import Audio as pyannAudio
# from pyannote.audio import Pipeline

# from pyannote.audio.pipelines.utils.hook import ProgressHook
#
# from IPython.display import Audio

from typing import List, NamedTuple, Optional, Tuple, Union, Dict, Generator
from datetime import datetime
import time
import pickle
from pathlib import Path

from transformers.feature_extraction_utils import PreTrainedFeatureExtractor, BatchFeature
from transformers.pipelines.base import no_collate_fn, pad_collate_fn
from transformers.pipelines.pt_utils import PipelineIterator, PipelineChunkIterator, PipelinePackIterator

from ..models.modeling_ct2_utils import Ct2PreTrainedModel
from utils.tokenizer import Tokenizer as fwTokenizer


def filter_segments(ann: Annotation, min_duration_on: float = 0.250, min_duration_off: float = 2.0) -> Annotation:
    """Filter segments based on their duration.

  Parameters
  ----------
  ann : Annotation
      Annotation to filter.

  min_duration_on : float, optional
      Remove speech regions shorter than that many seconds.
      Default is 0.250 seconds.

  min_duration_off : float, optional
      Fill gaps between speech regions shorter than many seconds.
      Default is 2.0 seconds.

  Returns
  -------
  Annotation
      Filtered annotation.

  """

    filtered = Annotation()

    # merge tracks having in between intervals shorter than min_duration_off
    if min_duration_off > 0.0:
        filtered = ann.support(collar=min_duration_off)

    # remove tracks shorter than min_duration_on
    if min_duration_on > 0:
        for segment, track in list(filtered.itertracks()):
            if segment.duration < min_duration_on:
                del filtered[segment, track]
    return filtered


# def get_prompt(
#         self,
#         tokenizer: Tokenizer,
#         previous_tokens: List[int],
#         without_timestamps: bool = False,
#         prefix: Optional[str] = None,
#         hotwords: Optional[str] = None,
# ) -> List[int]:
#     """
#     Эта функция из `transcribe.WhisperModel.get_prompt` аналогична по назначению
#     `transformers.models.whisper.generation_whisper.WhisperGenerationMixin._retrieve_init_tokens`
#
#     Пока использую ее как более простой воркэраунд, пока не напишу Ct2WhisperGenerationMixin
#
#     :param self:
#     :param tokenizer:
#     :param previous_tokens:
#     :param without_timestamps:
#     :param prefix:
#     :param hotwords:
#     :return:
#     """
#     prompt = []
#
#     if previous_tokens or (hotwords and not prefix):
#         prompt.append(tokenizer.sot_prev)
#         if hotwords and not prefix:
#             hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
#             if len(hotwords_tokens) >= self.max_length // 2:
#                 hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
#             prompt.extend(hotwords_tokens)
#         if previous_tokens:
#             prompt.extend(previous_tokens[-(self.max_length // 2 - 1):])
#
#     prompt.extend(tokenizer.sot_sequence)
#
#     if without_timestamps:
#         prompt.append(tokenizer.no_timestamps)
#
#     if prefix:
#         prefix_tokens = tokenizer.encode(" " + prefix.strip())
#         if len(prefix_tokens) >= self.max_length // 2:
#             prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
#         if not without_timestamps:
#             prompt.append(tokenizer.timestamp_begin)
#         prompt.extend(prefix_tokens)
#
#     return prompt


class Ct2Pipeline(Pipeline):
    """
    This abstract class aims to extend the base Hugging Face Pipeline to support CTranslate2 framework and models.
    The 4 methods the base class Pipeline preprocess, _forward, postprocess, and _sanitize_parameters are needed to be
    implemented in the successor class.
    """

    def __init__(self,
                 model: ctranslate2.models.Whisper,  #Ct2PreTrainedModel,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
                 framework: Optional[str] = "ct2",  # this is workaround to use correspondin code pf base Pipeline
                 device: Union[int, "torch.device"] = -1,
                 **kwargs,
                 ):
        self.model = model

        # workaround = copy-paste from WhisperX. Это работает, но нужно рефакторить, чтобы использовать из HF
        # self.tokenizer = tokenizer
        # self.options = options
        # self.preset_language = language
        # self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", 1)  # было дефолтное None
        self._num_workers = 0
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        # Чтобы инициализировать часть базового класса, нужно model привести к типу PreTrainedModel
        # Пока воркэраунд вообще не делать эту часть инициализации - как в WhisperX, где она не отрабатывает
        # super().__init__(self.model)

    def forward(self, model_inputs: dict[str, np.ndarray], **forward_params):
        """
            Override method of base class as we need add CTranslate2 specific for device handling.
            Reused and refactored from the `whisperx.asr.WhisperModel.encode` and
            `faster_whisper.transcribe.WhisperModel.encode`, as well from the base Hugging Face Pipeline class.

            :param model_inputs:
            :param forward_params:
            :return:

            """
        if self.framework == 'ct2':
            # When the model is running on multiple GPUs, the encoder output should be moved
            # to the CPU since we don't know which GPU will handle the next job.
            to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

            # if to_cpu:
            #     model_inputs['inputs'].to_device("cpu")

            # В WhisperX этот кусочек был в методе _forward, т.е. после формирования батча. Если я не ошибаюсь
            # unsqueeze if batch size = 1
            # if len(features['input_features'].shape) == 2:
            #     features['input_features'] = np.expand_dims(features['input_features'], 0)

            # Этот кусочек есть в faster-whisper, но нет
            # в примере https://opennmt.net/CTranslate2/guides/transformers.html#whisper
            # Проверить как будет работать, нет ли ошибки в типе данных.
            # Проверил. Вот это из документации не работает, ругаясь что у Tensor "Object does not implement the array interface":
            #     x = torch.ones((2, 4), dtype=torch.int32, device="cuda")
            #     y = ctranslate2.StorageView.from_array(x)
            # Поэтому оставляю как в faster-whisper.
            # model_inputs['input_features'] = np.ascontiguousarray(model_inputs['input_features'])

            inputs = ctranslate2.StorageView.from_array(model_inputs['input_features'])

            inputs = self.model.encode(inputs, to_cpu=to_cpu)

            model_inputs['input_features'] = inputs

            model_outputs = self._forward(model_inputs, **forward_params)
        else:
            raise ValueError(f"Framework {self.framework} is not supported by {self.__class__}")
            # super().forward(model_inputs['inputs'], **forward_params)
            # Выполнится это:
            # with self.device_placement():
            #     if self.framework == "tf":
            #         model_inputs["training"] = False
            #         model_outputs = self._forward(model_inputs, **forward_params)
            #     elif self.framework == "pt":
            #         inference_context = self.get_inference_context()
            #         with inference_context():
            #
            #             В WhisperX _ensure_tensor_on_device() не делает НИЧЕГО, т.к. model_inputs приходит
            #
            #             model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
            #             model_outputs = self._forward(model_inputs, **forward_params)
            #             model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
            #     else:
            #         raise ValueError(f"Framework {self.framework} is not supported")

        return model_outputs


class ASRoverSegmentationCt2Pipeline(Ct2Pipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio based on preliminarily
    made audio segmentation.

    The audio input can be either a raw waveform or an audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats

    The segmentation input can be pyannote Annotation object. This object is returned by a number of pyannote
    segmentation pipelines such as:
        - pyannote.audio.pipelines.voice_activity_detection
            .VoiceActivityDetection
            .AdaptiveVoiceActivityDetection
            .OracleVoiceActivityDetection
        - pyannote.audio.pipelines.speaker_diarization.SpeakerDiarization
        -

    """

    def __init__(self,
                 model: ctranslate2.models.Whisper,  #Ct2PreTrainedModel,
                 tokenizer: Optional[Union[WhisperTokenizer, WhisperTokenizerFast]] = None,
                 feature_extractor: Optional[WhisperFeatureExtractor] = None,
                 **kwargs
                 ):

        self.model = model
        self.tokenizer = tokenizer
        # self.fw_tokenizer = fwTokenizer(tokenizer, True, 'transcribe', tokenizer.language)
        self.feature_extractor = feature_extractor

        super().__init__(self.model,
                         self.tokenizer,
                         self.feature_extractor,
                         **kwargs)

    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        if args:
            print(f"Ignoring args : {args}")

        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size
        else:
            self._batch_size = batch_size

        # This is addition to HF Pipeline - the context which is available in all steps/processing functions
        # Like we may pass from preprocessing steps any data to postprocessing step and
        # at the same time no needs to pass them as argument through all Pipeline
        self._pipeline_context: Dict = {}
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        return self.get_iterator([inputs], num_workers, batch_size, preprocess_params, forward_params, postprocess_params)

    def get_iterator(
            self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):

        dataset = PipelineChunkIterator(inputs, self.segmenting, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            # logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        def _stack_dicts_collate_fn(items: List[Dict]) -> Dict[str, List]:
            return {"segments": [item["segment"] for item in items]}
        # TODO: проверить, почему где-то теряется последний сегмент, проявляется при разном размере батча
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,
                                collate_fn=_stack_dicts_collate_fn)  # Расчитываем получать просто List[np.ndarray]

        feature_iterator = PipelineIterator(dataloader, self.preprocess, preprocess_params)
        model_iterator = PipelineIterator(feature_iterator, self.forward, forward_params)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params, loader_batch_size=batch_size)
        return final_iterator

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "segmentation" in kwargs:
            preprocess_kwargs["segmentation"] = kwargs["segmentation"]
        if "min_duration_on" in kwargs:
            preprocess_kwargs["min_duration_on"] = kwargs["min_duration_on"]
        if "min_duration_off" in kwargs:
            preprocess_kwargs["min_duration_off"] = kwargs["min_duration_off"]

        return preprocess_kwargs, {}, {}

    def segmenting(self, audio: np.ndarray,
                   segmentation: Annotation = Annotation(),
                   min_duration_on=0.0,
                   min_duration_off=0.0,
                   **preprocess_parameters: Dict
                   ) -> Dict[str, np.ndarray]:
        """
        Use here standard Hugging Face Whisper Feature Extractor and Tokenizer to preprocess raw audio
        into Mel Spectrogram features.

        :param audio:
        :param segmentation:
        :param   min_duration_on : float, optional
                  Remove speech regions shorter than that many seconds.
                  Recommended is 0.250 seconds.

        :param   min_duration_off : float, optional
                  Fill gaps between speech regions shorter than many seconds.
                  Recommended is 2.0 seconds.
        :return:
        """

        # Filter and Merge segments based on speech duration and pauses between segments.
        segmentation_filtered = filter_segments(segmentation, min_duration_on, min_duration_off)
        # We need the new segmentation to match it with generated transcriptions on postprocess step
        self._pipeline_context['segmentation_filtered'] = segmentation_filtered

        for segment in segmentation_filtered.itersegments():
            start_sample = int(segment.start * self.feature_extractor.sampling_rate)
            end_sample = int(segment.end * self.feature_extractor.sampling_rate)
            audio_segment = audio[start_sample:end_sample]

            yield {'segment': audio_segment}

    def preprocess(self, audio_segments: Dict[str, List[np.ndarray]], **preprocess_parameters: Dict
                   ) -> Dict[str, np.ndarray]:
        """
        Extract features from audio data. The feature_extractor is applied on a batch of segments.

        :param audio_segments: Dict["segments", List[np.ndarray]], where list of segments is stacked by data_loader
        :param preprocess_parameters:
        :return: features as Dict["input_features", StorageView] object
        """
        features = self.feature_extractor(audio_segments["segments"],
                                          sampling_rate=self.feature_extractor.sampling_rate,
                                          return_tensors="np"  # np is required by ctranslate2.StorageView.from_array()
                                          )
        return {"input_features": features["input_features"]}

    def _forward(self, model_inputs: Dict[str, StorageView], **forward_parameters: Dict
                 ) -> Dict[str, List[WhisperGenerationResult]]:
        """
        Applying pure model.generate() and get text tokens.

        Note: ct2translate model's encoding is moved into the forward() method of Ct2Pipeline class, as we
        encapsulate there all functions concerning handling data on appropriate device.
        So we call the chain of forward()->_forward() method. See also get_iterator().

        :param model_inputs: Dict["input_features", StorageView]
        :return:
        """

        prompt = self.tokenizer.prefix_tokens
        # Here we catch the last batch, which might be shorter than `batch_size`
        effective_batch_size = model_inputs["input_features"].shape[0]
        result: List[WhisperGenerationResult] = self.model.generate(
                model_inputs["input_features"],
                [prompt] * effective_batch_size,
                # TODO: fill this parameters from Ct2GenerationMixin class
                # beam_size=options.beam_size,
                # patience=options.patience,
                # length_penalty=options.length_penalty,
                # max_length=self.max_length,
                # suppress_blank=options.suppress_blank,
                # suppress_tokens=options.suppress_tokens,
        )

        return {"generation_results_batch": result}

    def postprocess(self, model_outputs: Dict[str, List[WhisperGenerationResult]], **postprocess_parameters: Dict
                    ) -> Dict[str, List[Dict]]:
        """
        Here batch decoding happens. We work with token_ids here.
        Other generation results from WhisperGenerationResult do not processed currently

        :param model_outputs as Dict[str, List[WhisperGenerationResult]]
        :return: batch of transcriptions corresponding to audio segments as List[str]
        """
        tokens_batch = [x.sequences_ids[0] for x in model_outputs["generation_results_batch"]]
        texts = self.tokenizer.batch_decode(tokens_batch)

        timestamped_segments = []
        for text in texts:
            segment, track, label = next(self._pipeline_context["segmentation_filtered"].itertracks(yield_label=True))
            timestamped_segments.append({'text': text,
                                         'start': segment.start,
                                         'end': segment.end,
                                         'speaker': label
                                         })
            del self._pipeline_context["segmentation_filtered"][segment, track]
        return {'timestamped_segment': timestamped_segments}


