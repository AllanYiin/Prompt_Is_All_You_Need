from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import numpy as np
import soundfile as sf
import torch
import whisper
from whisper.utils import format_timestamp
from prompt4all import context
import pandas as pd

from typing import Optional, Union
import torch

cxt=context._context()


__all__ = ["whisper_model","to_formated_time","recognize_whisper","record_timeout","phrase_timeout","no_speech_threshold","load_whisper_model","load_audio","get_audio_duration"]

whisper_model=None
record_timeout=2
phrase_timeout=3
no_speech_threshold=0.6


class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker

def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'],seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker

        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(
                        diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'],word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker

    return transcript_result

def load_whisper_model():
    if cxt.whisper_model is None:
        cxt.whisper_model = whisper.load_model('medium', device='cpu')
        print('Whisper small model載入完成!')

    return cxt.whisper_model

def to_formated_time(float_time):
    return format_timestamp(float_time,always_include_hours=True)



def get_audio_duration(file: str):
    return float(ffmpeg.probe(file)["format"]["duration"])


def load_audio(file: str, sample_rate: int = 16000,
               start_time: str = None, duration: str = None):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Parameters
    ----------
    file: str
        The audio file to open
    sample_rate: int
        The sample rate to resample the audio if necessary
    start_time: str
        The start time, using the standard FFMPEG time duration syntax, or None to disable.

    duration: str
        The duration, using the standard FFMPEG time duration syntax, or None to disable.
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        inputArgs = {'threads': 0}

        if (start_time is not None):
            inputArgs['ss'] = start_time
        if (duration is not None):
            inputArgs['t'] = duration

        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, **inputArgs)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}")

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def recognize_whisper(audio_data,word_timestamps=False,language='zh', translate=False, **transcribe_options):
    """
    Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using Whisper.

    The recognition language is determined by ``language``, an uncapitalized full language name like "english" or "chinese". See the full language list at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

    model can be any of tiny, base, small, medium, large, tiny.en, base.en, small.en, medium.en. See https://github.com/openai/whisper for more details.

    If show_dict is true, returns the full dict response from Whisper, including the detected language. Otherwise returns only the transcription.

    You can translate the result to english with Whisper by passing translate=True

    Other values are passed directly to whisper. See https://github.com/openai/whisper/blob/main/whisper/transcribe.py for all options
    """

    if cxt.whisper_model is None:
        cxt.whisper_model=whisper.load_model('small', device='cuda')
        return {"text":'Whisper medium model載入完成!',"no_speech_prob":0.001}

    # 16 kHz https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/audio.py#L98-L99
    if not isinstance(audio_data,np.ndarray):
        # wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        # wav_stream = io.BytesIO(wav_bytes)
        # audio_array, sampling_rate = sf.read(wav_stream)
        # audio_array = audio_array.astype(np.float32)

        result = cxt.whisper_model.transcribe(
            audio_data,
            language=language,
            word_timestamps=word_timestamps,
            verbose=True,
            task="translate" if translate else None,
            fp16=True if cxt.whisper_model.device=="cuda" and torch.cuda.is_available() else False,
            no_speech_threshold=0.65,
            initial_prompt="#zh-tw 使用ChatGPT以及Whisper會議記錄逐字稿",
            **transcribe_options
        )

    else:
        audio_array=audio_data.astype(np.float16  if cxt.whisper_model.device=="cuda" and torch.cuda.is_available() else np.float32 )
        result = cxt.whisper_model.transcribe(
            audio_array,
            language=language,
            word_timestamps=word_timestamps,
            task="translate" if translate else None,
            fp16=True if cxt.whisper_model.device=="cuda" and torch.cuda.is_available() else False,
            no_speech_threshold=0.6,
            initial_prompt="#zh-tw 使用ChatGPT以及Whisper會議記錄逐字稿",
            **transcribe_options
        )

    return result






