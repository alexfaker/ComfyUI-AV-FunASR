import jieba
import string
import json
import Levenshtein
import re
import nltk
from nltk.tokenize import word_tokenize
import nltk.data

class PunctuationConfig:
    """统一管理标点符号配置"""
    
    # 完整的标点符号集合
    CHINESE_PUNCTUATION = "，。；！？：""''（）【】《》〈〉「」『』〔〕…—、·‧丶"
    ENGLISH_PUNCTUATION = ",.;!?:\"'()[]{}<>/-\\|~`@#$%^&*+=_"
    MATH_SYMBOLS = "×÷±∞∫∑∏√∂∇∆∅∪∩⊂⊃⊄⊅"
    OTHER_SYMBOLS = "•◦‣⁃▪▫▸▹▽△▼▲◇◆○●◯◉"
    
    # 所有标点符号
    ALL_PUNCTUATION = CHINESE_PUNCTUATION + ENGLISH_PUNCTUATION + MATH_SYMBOLS + OTHER_SYMBOLS
    
    # 断句标点符号（用于分词）
    SENTENCE_BREAK_PUNCTUATION = "，。；！？：""''（）【】《》〈〉「」『』〔〕…—、·‧丶,.;!?:\"'()[]{}<>/-\\|"
    
    # 标点符号优先级映射
    PUNCTUATION_PRIORITY = {
        # 强断句符号 (优先级1-2)
        '。': 1, '.': 1,
        '？': 2, '?': 2,
        '！': 3, '!': 3,
        
        # 中断句符号 (优先级3-4)
        '；': 4, ';': 4,
        '：': 5, ':': 5,
        '…': 6, '—': 6,
        
        # 弱断句符号 (优先级5-6)
        '，': 7, ',': 7,
        '、': 8, '·': 8, '‧': 8,
        
        # 配对符号 (优先级7-8)
        '"': 9, '"': 9, "'": 9, "'": 9, '"': 9, "'": 9,
        '（': 10, '）': 10, '(': 10, ')': 10,
        '【': 11, '】': 11, '[': 11, ']': 11,
        '《': 12, '》': 12, '<': 12, '>': 12,
        '〈': 13, '〉': 13, '「': 13, '」': 13,
        '『': 14, '』': 14, '〔': 14, '〕': 14,
        '{': 15, '}': 15,
        
        # 其他符号 (优先级较低)
        '/': 16, '\\': 16, '|': 16, '-': 16, '_': 16,
        '~': 17, '`': 17, '@': 17, '#': 17, '$': 17, '%': 17, '^': 17, '&': 17, '*': 17, '+': 17, '=': 17,
    }
    
    @classmethod
    def get_priority(cls, char):
        """获取标点符号优先级，数字越小优先级越高"""
        return cls.PUNCTUATION_PRIORITY.get(char, 999)
    
    @classmethod
    def is_punctuation(cls, char):
        """检查字符是否为标点符号"""
        return char in cls.ALL_PUNCTUATION
    
    @classmethod
    def is_sentence_break(cls, char):
        """检查字符是否为断句标点符号"""
        return char in cls.SENTENCE_BREAK_PUNCTUATION

class Format2Subtitle:
    def __init__(self, asr_result, ori_text):
        self.asr_result = asr_result
        self.ori_text = ori_text
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """确保NLTK数据包已下载"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("正在下载NLTK数据包...")
            nltk.download('punkt', quiet=True)
    
    def detect_language(self, text):
        """
        检测文本语言类型
        
        参数:
            text (str): 待检测的文本
            
        返回:
            str: 'chinese', 'english', 'mixed'
        """
        if not text:
            return 'chinese'  # 默认为中文
        
        # 统计中文字符数量
        chinese_count = 0
        english_count = 0
        total_chars = 0
        
        for char in text:
            if char.isspace() or PunctuationConfig.is_punctuation(char):
                continue
            total_chars += 1
            
            # 检查是否为中文字符（Unicode范围）
            if '\u4e00' <= char <= '\u9fff':
                chinese_count += 1
            # 检查是否为英文字符
            elif char.isalpha() and ord(char) < 128:
                english_count += 1
        
        if total_chars == 0:
            return 'chinese'  # 默认为中文
        
        chinese_ratio = chinese_count / total_chars
        english_ratio = english_count / total_chars
        
        # 判断语言类型
        if chinese_ratio > 0.5:
            return 'chinese'
        elif english_ratio > 0.5:
            return 'english'
        else:
            return 'mixed'
    
    def tokenize_text(self, text, language='auto'):
        """
        根据语言类型进行分词
        
        参数:
            text (str): 待分词的文本
            language (str): 语言类型，'auto'为自动检测，'chinese'为中文，'english'为英文
            
        返回:
            list: 分词结果列表
        """
        if not text:
            return []
        
        # 自动检测语言
        if language == 'auto':
            language = self.detect_language(text)
        
        # 根据语言类型选择分词器
        if language == 'chinese':
            return jieba.lcut(text)
        elif language == 'english':
            try:
                tokens = word_tokenize(text)
                return tokens
            except Exception as e:
                print(f"英文分词失败，使用空格分词: {e}")
                return text.split()
        elif language == 'mixed':
            # 混合语言处理：先用jieba分词，再对英文部分进行细分
            chinese_tokens = jieba.lcut(text)
            final_tokens = []
            
            for token in chinese_tokens:
                # 检查token是否主要包含英文
                if self._is_english_token(token):
                    try:
                        english_subtokens = word_tokenize(token)
                        final_tokens.extend(english_subtokens)
                    except Exception:
                        final_tokens.append(token)
                else:
                    final_tokens.append(token)
            
            return final_tokens
        else:
            # 默认使用中文分词
            return jieba.lcut(text)
    
    def _is_english_token(self, token):
        """检查token是否主要包含英文字符"""
        if not token:
            return False
        
        english_count = 0
        total_chars = 0
        
        for char in token:
            if char.isspace() or PunctuationConfig.is_punctuation(char):
                continue
            total_chars += 1
            if char.isalpha() and ord(char) < 128:
                english_count += 1
        
        return total_chars > 0 and english_count / total_chars > 0.5  
    def ms_to_srt_time(self, ms):
        """
        将毫秒转换为SRT时间格式 (HH:MM:SS,mmm)
        例如：150毫秒 -> 00:00:00,150
        """
        hours = ms // 3600000  # 1小时 = 3,600,000毫秒
        ms %= 3600000
        minutes = ms // 60000  # 1分钟 = 60,000毫秒
        ms %= 60000
        seconds = ms // 1000   # 1秒 = 1,000毫秒
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    # 查询标点符号位置
    def find_punctuation_indices(self, words):
        punctuation = """，。；！？!,;.?:"""
        for i, word in enumerate(words):
            if word in punctuation:
                return i
        return -1

    def get_punctuation_priority(self, char):
        """获取标点符号优先级，数字越小优先级越高"""
        return PunctuationConfig.get_priority(char)

    def find_best_punctuation_index(self, words, min_length=3, max_length=15):
        """在指定范围内查找最佳断句位置"""
        best_index = -1
        best_priority = 999
        
        # 在合理范围内寻找标点符号
        search_end = min(len(words), max_length)
        for i in range(min_length, search_end):
            if i < len(words):
                word = words[i]
                # 使用PunctuationConfig检查是否为断句标点符号
                if PunctuationConfig.is_sentence_break(word):
                    priority = PunctuationConfig.get_priority(word)
                    if priority < best_priority:
                        best_priority = priority
                        best_index = i
        
        return best_index

    # 清除句子中的断句符号
    def remove_sentence_punctuation(self, sentence):
        """去除句子中的断句符号"""
        # 使用统一的标点符号配置
        for p in PunctuationConfig.SENTENCE_BREAK_PUNCTUATION:
            sentence = sentence.replace(p, ' ')
        sentence = sentence.strip()
        return sentence

    def align_char_timestamps(self, original_sentence, recognized_text, word_timestamps, text_length=12):
        """
        将词语级时间戳对齐到字符级时间戳
        
        参数:
            original_sentence (str): 带标点的原始句子
            recognized_text (str): 空格分隔的无标点识别文本
            word_timestamps (list): 词语级时间戳列表 [[start1, end1], [start2, end2], ...]
            
        返回:
            list: 原始字符级时间戳列表 [[char_start, char_end], ...]
        """
        if not original_sentence:
            original_sentence = "".join(recognized_text.split())

        # 使用新的多语言分词接口
        words = self.tokenize_text(original_sentence, language='auto')

        print(len(original_sentence), len(word_timestamps), len(recognized_text))
        recognized_words = recognized_text.split()

        # 4. 为每个字符分配时间戳
        char_timestamps = []
        
        while True:
            if len(words) < 3:  # 调整最小长度要求
                break
            # 查询列表中的最佳标点符号断句位置
            punctuation_indices = self.find_best_punctuation_index(words, min_length=3, max_length=15)
            print("最佳断句位置：", punctuation_indices)
            
            # 如果找到了合适的标点符号位置，使用标点符号断句
            if punctuation_indices > 0:
                _words = words[:punctuation_indices + 1]  # 包含标点符号
                words = words[punctuation_indices + 1:]  # 跳过标点符号
            else:
                # 如果没有找到合适的标点符号，使用默认长度分割
                split_length = min(text_length, len(words))
                _words = words[:split_length]
                words = words[split_length:]

            text = ''.join(_words)
            # 计算词语持续时间
            best_index = -1
            best_score = 9999
            for index in range(text_length*4):
                _recognized_words = recognized_words[:index]
                _recognized_text = ''.join(_recognized_words)
                if text == _recognized_text:
                    break
                # 计算编辑距离
                distance = Levenshtein.distance(text, _recognized_text)
                if distance < best_score:
                    best_index = index
                    best_score = distance
            _recognized_words = recognized_words[:best_index]
            recognized_words = recognized_words[best_index:]

            timestamp = word_timestamps[:best_index]
            word_timestamps = word_timestamps[best_index:]
            word_start = timestamp[0][0]
            word_end = timestamp[-1][1]
            
            char_timestamps.append([word_start, word_end, self.remove_sentence_punctuation(text)])
        # 处理最后一段
        if len(words) > 0 and len(word_timestamps) > 0:
            text = ''.join(words)
            timestamp = word_timestamps
            word_start = timestamp[0][0]
            word_end = timestamp[-1][1]
            char_timestamps.append([word_start, word_end, self.remove_sentence_punctuation(text)])
        return char_timestamps

    def format_subtitle(self, char_timestamps):
        """
        [230, 2210, '一个能够从音频中生成面部']
        [230, 5430, '运动系数的工具']
        转为字幕文件
        """
        srt_content = ""
        counter = 1
        for item in char_timestamps:
            start_time = self.ms_to_srt_time(item[0])
            end_time = self.ms_to_srt_time(item[1])
            text = item[2]
            srt_content += f"{counter}\n{start_time} --> {end_time}\n{text}\n\n"
            # print(srt_content)
            counter += 1
        return srt_content
    
    def pipeline(self, output=None):
        # print(rec_result)
        recognized_text = self.asr_result['text']
        word_timestamps = self.asr_result['timestamp']
        original_sentence = self.ori_text
        char_timestamps = self.align_char_timestamps(original_sentence, recognized_text, word_timestamps)
        subtitle_text = self.format_subtitle(char_timestamps)
        if output is not None:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(subtitle_text)
        return subtitle_text