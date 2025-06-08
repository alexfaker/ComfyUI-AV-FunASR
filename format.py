import jieba
import string
import json
import Levenshtein
import re

class Format2Subtitle:
    def __init__(self, asr_result, ori_text):
        self.asr_result = asr_result
        self.ori_text = ori_text  
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

    # 清除句子中的断句符号
    def remove_sentence_punctuation(self, sentence):
        """去除句子中的断句符号"""
        punctuation = """，。；！？!,;.?:()（）"""
        for p in punctuation:
            sentence = sentence.replace(p, ' ')
        sentence = sentence.strip()
        return sentence

    def align_char_timestamps(self, original_sentence, recognized_text, word_timestamps, text_length=8):
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

        words = jieba.lcut(original_sentence)

        # print(len(original_sentence), len(word_timestamps), len(recognized_text))
        recognized_words = recognized_text.split()

        # 4. 为每个字符分配时间戳
        char_timestamps = []
        
        while True:
            if len(words) < 10:
                break
            # 查询列表中的标点符号的位置
            punctuation_indices = self.find_punctuation_indices(words)
            # print("标点符号位置：",punctuation_indices)
            # 如果位置小于10，怎使用这个位置分割
            if punctuation_indices > 10 or punctuation_indices <= 2:
                _words = words[:text_length]
                words = words[text_length:]
            else:
                _words = words[:punctuation_indices]
                words = words[punctuation_indices:]

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
        if  len(word_timestamps) > 0:
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