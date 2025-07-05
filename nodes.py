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
from PIL import Image, ImageOps
import requests
import io
import numpy as np

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


class AVLoadImageFromURL:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": "", "tooltip": "图片文件的URL链接"}),
                "timeout": ("INT", {"default": 30, "min": 5, "max": 300, "step": 1, "tooltip": "下载超时时间（秒）"}),
            },
            "optional": {
                "max_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64, "tooltip": "最大尺寸限制，0为不限制"}),
                "cache_enabled": ("BOOLEAN", {"default": True, "tooltip": "是否启用缓存"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "load_image_from_url"
    CATEGORY = "Aven/AV-FunASR"
    DESCRIPTION = "从URL链接下载并展示图片"

    def load_image_from_url(self, url, timeout=30, max_size=0, cache_enabled=True):
        try:
            # 验证URL格式
            if not url or not url.startswith(('http://', 'https://')):
                raise ValueError("请提供有效的HTTP或HTTPS URL")
            
            
            # 设置多个User-Agent选项用于重试
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'
            ]
            
            last_error = None
            for i, user_agent in enumerate(user_agents):
                try:
                    headers = {
                        'User-Agent': user_agent,
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                    
                    print(f"正在从URL下载图片 (尝试 {i+1}/{len(user_agents)}): {url}")
                    
                    with requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as response:
                        response.raise_for_status()
                        
                        # 检查内容类型
                        content_type = response.headers.get('content-type', '').lower()
                        print(f"图片文件类型: {content_type}")
                        
                        # 验证是否为图片类型（检查content-type和文件扩展名）
                        is_image_content_type = content_type.startswith('image/')
                        is_image_extension = self._is_image_url(url)
                        
                        if not is_image_content_type and not is_image_extension:
                            error_details = self._get_error_details(response, content_type)
                            error_msg = f"URL返回的不是图片类型，content-type: {content_type}，文件扩展名也不是图片格式\n{error_details}"
                            if i == len(user_agents) - 1:  # 最后一次尝试
                                raise ValueError(error_msg)
                            else:
                                last_error = error_msg
                                print(f"尝试 {i+1} 失败，使用不同的User-Agent重试...")
                                import time
                                time.sleep(1)  # 短暂延时
                                continue
                        elif not is_image_content_type and is_image_extension:
                            print(f"警告: content-type不是图片格式({content_type})，但文件扩展名是图片格式，继续处理...")
                        
                        # 获取图片数据
                        image_data = response.content
                        print(f"图片下载完成，大小: {len(image_data)} bytes")
                        
                        # 使用PIL加载图片
                        image = Image.open(io.BytesIO(image_data))
                        
                        # 处理图片格式
                        original_format = image.format
                        original_size = image.size
                        print(f"原始图片格式: {original_format}, 尺寸: {original_size}")
                        
                        # 转换为RGB格式（如果需要）
                        if image.mode != 'RGB':
                            if image.mode == 'RGBA':
                                # 处理透明背景，使用白色背景
                                background = Image.new('RGB', image.size, (255, 255, 255))
                                background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
                                image = background
                            else:
                                image = image.convert('RGB')
                        
                        # 应用尺寸限制
                        if max_size > 0:
                            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                            print(f"图片已调整尺寸: {image.size}")
                        
                        # 转换为ComfyUI所需的tensor格式
                        image_tensor = self._convert_to_tensor(image)
                        
                        width, height = image.size
                        print(f"图片加载成功: 尺寸={width}x{height}, tensor形状={image_tensor.shape}")
                        
                        return (image_tensor, width, height)
                        
                except requests.exceptions.RequestException as e:
                    last_error = str(e)
                    if i == len(user_agents) - 1:  # 最后一次尝试
                        raise
                    print(f"尝试 {i+1} 失败: {last_error}，使用不同的User-Agent重试...")
                    import time
                    time.sleep(1)
                    continue
                    
        except requests.exceptions.Timeout:
            raise RuntimeError(f"下载超时: URL={url}, 超时时间={timeout}秒")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"加载图片失败: {str(e)}")
    
    def _convert_to_tensor(self, image):
        """将PIL图片转换为ComfyUI张量格式"""
        # 转换为numpy数组
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # 确保是3通道RGB格式
        if len(image_np.shape) == 2:
            # 灰度图转RGB
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:
            # RGBA转RGB (这里应该已经在上面处理过了，但以防万一)
            image_np = image_np[:, :, :3]
        
        # 转换为PyTorch张量，格式为 [batch, height, width, channels]
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        return image_tensor
     
    def _get_error_details(self, response, content_type):
        """获取详细的错误信息"""
        try:
            error_details = []
            
            # 添加HTTP状态码
            error_details.append(f"HTTP状态码: {response.status_code}")
            
            # 添加响应头信息
            useful_headers = ['server', 'content-length', 'last-modified', 'location']
            for header in useful_headers:
                if header in response.headers:
                    error_details.append(f"{header}: {response.headers[header]}")
            
            # 如果是HTML页面，尝试获取页面内容预览
            if 'text/html' in content_type:
                try:
                    content_preview = response.text[:200]  # 前200字符
                    if content_preview:
                        # 清理HTML标签，获取纯文本
                        import re
                        clean_text = re.sub(r'<[^>]+>', '', content_preview)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        if clean_text:
                            error_details.append(f"页面内容预览: {clean_text}")
                except:
                    pass
            
            # 添加解决建议
            suggestions = []
            if 'text/html' in content_type:
                suggestions.append("• 检查URL是否正确，可能需要直接访问图片文件而不是网页")
                suggestions.append("• 尝试在浏览器中直接访问URL查看实际内容")
                suggestions.append("• 如果是私有链接，可能需要身份验证")
            
            if suggestions:
                error_details.append("解决建议:")
                error_details.extend(suggestions)
            
            return "\n".join(error_details)
            
        except Exception as e:
            return f"无法获取详细错误信息: {str(e)}"
    
    def _is_image_url(self, url):
        """检查URL的文件扩展名是否为图片格式"""
        # 常见的图片文件扩展名
        image_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', 
            '.webp', '.svg', '.ico', '.avif', '.heic', '.heif'
        }
        
        try:
            # 移除查询参数，只检查文件路径
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # 检查文件扩展名
            for ext in image_extensions:
                if path.endswith(ext):
                    return True
            
            return False
        except:
            return False

    

NODE_CLASS_MAPPINGS = {
    "AVSpeechTimestamp": AVSpeechTimestamp,
    "AVASRTimestamp": AVASRTimestamp,
    "AVFormat2Subtitle": AVFormat2Subtitle,
    "AVSaveSubtitles": AVSaveSubtitles,
    "AVLoadAudioFromURL": AVLoadAudioFromURL,
    "AVLoadImageFromURL": AVLoadImageFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AVSpeechTimestamp": "AV Speech Timestamp",
    "AVASRTimestamp": "AV ASR Timestamp",
    "AVFormat2Subtitle": "AV Format to Subtitle",
    "AVSaveSubtitles": "AV Save Subtitles",
    "AVLoadAudioFromURL": "AV Load Audio From URL",
    "AVLoadImageFromURL": "AV Load Image From URL",
}