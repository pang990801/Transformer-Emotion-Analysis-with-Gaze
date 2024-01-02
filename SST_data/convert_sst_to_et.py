import csv
import nltk
import re
from nltk.tokenize import word_tokenize

# 下载 NLTK 的分词器数据（如果尚未下载）
nltk.download('punkt')

def process_sst_file(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, \
            open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 写入表头
        writer.writerow(['sentence_id', 'word_id', 'word', 'nFix', 'FFD', 'GPT', 'TRT', 'GD'])

        sentence_id = 0
        for row in reader:
            sentence = row[0]
            # 使用 NLTK 的 word_tokenize 进行分词
            words = word_tokenize(sentence)

            # 使用正则表达式过滤，只保留字母组成的单词
            words = [word for word in words if re.match("^[A-Za-z]+$", word)]

            # 如果没有单词，添加一个占位符
            if not words:
                words = ["unknown"]

            for word_id, word in enumerate(words):
                writer.writerow([sentence_id, word_id, word, 0, 0, 0, 0, 0])

            sentence_id += 1

# 使用函数
input_csv = 'stts_all_sentence_level.csv'  # 替换为你的输入 CSV 文件路径
output_csv = 'sst_et_test.csv'  # 替换为你希望的输出 CSV 文件路径
process_sst_file(input_csv, output_csv)
