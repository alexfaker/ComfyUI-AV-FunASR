import folder_paths
import os
import comfy.model_management as mm
import time
import torchaudio
import torchvision.utils as vutils
import torch
import json
import uuid
from comfy.comfy_types import FileLocator

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
            # vad_model = os.path.join(model_root, name_maps_ms["fsmn-vad"])
            
            os.makedirs(model_dir, exist_ok=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            AVSpeechTimestamp.infer_ins_cache = AutoModel(
                model=model_dir,
                model_revision="v2.0.4",
                device=device,  # GPU加速
                disable_update=True
            )

        # save audio
        uuidv4 = str(uuid.uuid4())
        audio_save_path = os.path.join(temp_dir, f"{uuidv4}.wav")
        waveform = audio['waveform']
        sr = audio["sample_rate"]
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        torchaudio.save(audio_save_path, waveform.squeeze(0), 16000)

        rec_result = AVSpeechTimestamp.infer_ins_cache.generate(
            input=(audio_save_path, text), 
            data_type=("sound", "text"),
        )
        # print(rec_result)
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
        # print(text, jr)
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
                device=device,  # GPU加速
                disable_update=True
            )
        # save 
        uuidv4 = str(uuid.uuid4())
        audio_save_path = os.path.join(temp_dir, f"{uuidv4}.wav")
        # 重新采样为16k
        waveform = audio['waveform']
        sr = audio["sample_rate"]
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        torchaudio.save(audio_save_path, waveform.squeeze(0), 16000)

        rec_result = AVASRTimestamp.infer_ins_cache.generate(input=audio_save_path, batch_size_s=batch_size_s)
        # print(rec_result)
        if rec_result:
            rec_result = rec_result[0]

        # infer
        if unload_model:
            import gc
            if AVASRTimestamp.infer_ins_cache is not None:
                AVASRTimestamp.infer_ins_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                print("AVASRTimestamp memory cleanup successful")
        # jr = json.dumps(rec_result, indent=4)
        text = rec_result.get("text")
        jr = json.dumps(rec_result, ensure_ascii=False)
        # print((text, jr, rec_result))
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
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results: list[FileLocator] = []

        file = f"{filename}_{counter:05}_.srt"
        with open(os.path.join(full_output_folder, file), 'w', encoding='utf-8') as f:
            f.write(subtitles)
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

        return { "ui": { "subtitles": results } }


class AVLoadAudioFromURL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": "", "tooltip": "音频文件的URL链接"}),
                "timeout": ("INT", {"default": 30, "min": 5, "max": 300, "step": 1, "tooltip": "下载超时时间（秒）"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio_from_url"
    CATEGORY = "Aven/AV-FunASR"
    DESCRIPTION = "从URL链接加载音频文件"

    def load_audio_from_url(self, url, timeout=30):
        import requests
        import io
        import tempfile
        
        try:
            # 验证URL格式
            if not url or not url.startswith(('http://', 'https://')):
                raise ValueError("请提供有效的HTTP或HTTPS URL")
            
            # 设置请求头，模拟浏览器访问
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # 下载音频文件
            print(f"正在从URL下载音频: {url}")
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                response.raise_for_status()
                
                # 检查内容类型
                content_type = response.headers.get('content-type', '')
                print(f"音频文件类型: {content_type}")
                
                # 获取音频数据
                audio_data = response.content
                
                # 创建临时文件
                temp_dir = folder_paths.get_temp_directory()
                os.makedirs(temp_dir, exist_ok=True)
                
                # 根据URL或content-type确定文件扩展名
                file_extension = '.wav'  # 默认扩展名
                if 'audio/mpeg' in content_type or url.lower().endswith('.mp3'):
                    file_extension = '.mp3'
                elif 'audio/wav' in content_type or url.lower().endswith('.wav'):
                    file_extension = '.wav'
                elif 'audio/flac' in content_type or url.lower().endswith('.flac'):
                    file_extension = '.flac'
                elif 'audio/x-m4a' in content_type or url.lower().endswith('.m4a'):
                    file_extension = '.m4a'
                elif url.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac')):
                    file_extension = '.' + url.split('.')[-1].lower()
                
                # 保存临时文件
                temp_filename = f"url_audio_{uuid.uuid4().hex}{file_extension}"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                
                print(f"音频文件已保存到临时路径: {temp_path}")
                
                # 使用torchaudio加载音频
                waveform, sample_rate = torchaudio.load(temp_path)
                
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                print(f"音频加载成功: 采样率={sample_rate}, 波形形状={waveform.shape}")
                
                # 返回标准的AUDIO格式
                return ({"waveform": waveform, "sample_rate": sample_rate},)
                
        except requests.exceptions.Timeout:
            raise RuntimeError(f"下载超时: URL={url}, 超时时间={timeout}秒")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"加载音频失败: {str(e)}")

    

NODE_CLASS_MAPPINGS = {
    "AVSpeechTimestamp": AVSpeechTimestamp,
    "AVASRTimestamp": AVASRTimestamp,
    "AVFormat2Subtitle": AVFormat2Subtitle,
    "AVSaveSubtitles": AVSaveSubtitles,
    "AVLoadAudioFromURL": AVLoadAudioFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AVSpeechTimestamp": "AV Speech Timestamp",
    "AVASRTimestamp": "AV ASR Timestamp",
    "AVFormat2Subtitle": "AV Format to Subtitle",
    "AVSaveSubtitles": "AV Save Subtitles",
    "AVLoadAudioFromURL": "AV Load Audio From URL",
}