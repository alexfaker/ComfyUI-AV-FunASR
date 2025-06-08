import folder_paths
import os
import comfy.model_management as mm
import time
import torchaudio
import torchvision.utils as vutils
import torch
import json

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from funasr import AutoModel
from .format import Format2Subtitle



name_maps_ms = {
    "paraformer": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-en": "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
    "paraformer-en-spk": "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
    "paraformer-zh-streaming": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    "ct-punc-c": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "fa-zh": "iic/speech_timestamp_prediction-v1-16k-offline",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
    "Whisper-large-v2": "iic/speech_whisper-large_asr_multilingual",
    "Whisper-large-v3": "iic/Whisper-large-v3",
    "Qwen-Audio": "Qwen/Qwen-Audio",
    "emotion2vec_plus_large": "iic/emotion2vec_plus_large",
    "emotion2vec_plus_base": "iic/emotion2vec_plus_base",
    "emotion2vec_plus_seed": "iic/emotion2vec_plus_seed",
    "Whisper-large-v3-turbo": "iic/Whisper-large-v3-turbo",
}

class AVSpeechTimestamp:
    infer_ins_cache = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING",),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "ASRRESULT")
    RETURN_NAMES = ("text", "json_result", "asr_result")
    FUNCTION = "infer"
    CATEGORY = "Aven/AV-FunASR"
    DESCRIPTION = "get speech timestamp"

    def infer(self, audio, text, unload_model):
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        if AVSpeechTimestamp.infer_ins_cache is None:
            model_root = os.path.join(folder_paths.models_dir, "ASR/FunASR")
            model_dir = os.path.join(model_root, name_maps_ms["fa-zh"])
            
            os.makedirs(model_dir, exist_ok=True)
            AVSpeechTimestamp.infer_ins_cache = pipeline(
                task=Tasks.speech_timestamp,
                model=model_dir,
                model_revision="v2.0.4",
                output_dir=temp_dir
            )
        # save audio
        audio_save_path = os.path.join(temp_dir, f"{int(time.time())}.wav")
        torchaudio.save(audio_save_path, audio['waveform'].squeeze(0), audio["sample_rate"])

        rec_result = AVSpeechTimestamp.infer_ins_cache(input=(audio_save_path,text), data_type=("sound", "text"))
        if rec_result:
            rec_result = rec_result[0]

        # infer
        if unload_model:
            import gc
            if AVSpeechTimestamp.infer_ins_cache is not None:
                AVSpeechTimestamp.infer_ins_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                print("AVSpeechTimestamp memory cleanup successful")
        # jr = json.dumps(rec_result, indent=4)
        text = rec_result.get("text")
        jr = json.dumps(rec_result, ensure_ascii=False)
        print(text, jr)
        return (text, jr, rec_result)
    

class AVASRTimestamp:
    infer_ins_cache = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "batch_size_s": ("INT", {"default": 300, "min": 30, "max": 300, "step": 1}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "ASRRESULT")
    RETURN_NAMES = ("text", "json_result", "asr_result")
    FUNCTION = "infer"
    CATEGORY = "Aven/AV-FunASR"
    DESCRIPTION = "get speech timestamp"

    def infer(self, audio, batch_size_s, unload_model):
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        if AVASRTimestamp.infer_ins_cache is None:
            model_root = os.path.join(folder_paths.models_dir, "ASR/FunASR")
            model_dir = os.path.join(model_root, name_maps_ms["paraformer-zh"])
            vad_model = os.path.join(model_root, name_maps_ms["fsmn-vad"])
            os.makedirs(model_dir, exist_ok=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            AVASRTimestamp.infer_ins_cache = AutoModel(
                model=model_dir,
                vad_model=vad_model,
                punc_model=None, #"ct-punc",
                device=device  # GPU加速
            )
        # save audio
        audio_save_path = os.path.join(temp_dir, f"{int(time.time())}.wav")
        torchaudio.save(audio_save_path, audio['waveform'].squeeze(0), audio["sample_rate"])

        rec_result = AVASRTimestamp.infer_ins_cache.generate(input=audio_save_path, batch_size_s=batch_size_s)
        if rec_result:
            rec_result = rec_result[0]

        # infer
        if unload_model:
            import gc
            if AVSpeechTimestamp.infer_ins_cache is not None:
                AVSpeechTimestamp.infer_ins_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                print("AVSpeechTimestamp memory cleanup successful")
        # jr = json.dumps(rec_result, indent=4)
        text = rec_result.get("text")
        jr = json.dumps(rec_result, ensure_ascii=False)
        print((text, jr, rec_result))
        return (text, jr, rec_result)
    

class AVFormat2Subtitle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "asr_result": ("ASRRESULT",),
                "text":  ("STRING", {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("subtitle", )
    FUNCTION = "format_subtitle"
    CATEGORY = "Aven/AV-FunASR"
    DESCRIPTION = "format asr result to subtitle"

    def format_subtitle(self, asr_result, text=None):
        f = Format2Subtitle(asr_result, ori_text=text)
        content = f.pipeline()
        
        return (content, )
    

class AVSaveSubtitles:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subtitles": ("STRING", {"tooltip": "The subtitles to save."}),
                "filename_prefix": ("STRING", {"default": "subtitles", "tooltip": "The prefix for the file to save. "})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_subtitles"

    OUTPUT_NODE = True

    CATEGORY = "Aven/AV-FunASR"
    DESCRIPTION = "Saves the subtitles to a file."

    def save_subtitles(self, subtitles, filename_prefix="subtitles"):
        filename = f"{filename_prefix}-{time.time()}.srt"
        save_path = os.path.join(self.output_dir, filename)
        with open(save_path, "w") as f:
            f.write(subtitles)

        return { "ui": { "text": save_path } }


    

NODE_CLASS_MAPPINGS = {
    "AVSpeechTimestamp": AVSpeechTimestamp,
    "AVASRTimestamp": AVASRTimestamp,
    "AVFormat2Subtitle": AVFormat2Subtitle,
    "AVSaveSubtitles": AVSaveSubtitles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AVSpeechTimestamp": "AV Speech Timestamp",
    "AVASRTimestamp": "AV ASR Timestamp",
    "AVFormat2Subtitle": "AV Format to Subtitle",
    "AVSaveSubtitles": "AV Save Subtitles",
}