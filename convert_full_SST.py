import os
import pandas as pd

sentiment_mapping = {
    'NEGATIVE': 0,
    'POSITIVE': 2,
    'NEUTRAL': 1
}

data = []

for folder in ['NEGATIVE', 'POSITIVE', 'NEUTRAL']:
    dir_path = os.path.join('ZuCo_SST_data/all', folder)
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            sentence_id = int(os.path.splitext(filename)[0])
            data.append([sentence_id, content, sentiment_mapping[folder]])

df = pd.DataFrame(data, columns=['sentence_id', 'sentence', 'sentiment_label'])

df = df.sort_values(by='sentence_id')

df.to_csv('ZuCo_SST_data/ssts_ZuCo.csv', index=False)