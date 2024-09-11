import pickle

from word2sequence import Word2Sequece
import os
from dataset import tokenlize
from tqdm import tqdm

if __name__ == '__main__':
    ws = Word2Sequece()
    path = '../../dataset/aclImdb/train'
    temp_data_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    for data_path in temp_data_path:
        file_paths = [os.path.join(data_path, i) for i in os.listdir(data_path) if i.endswith('.txt')]
        for file_path in tqdm(file_paths):
            sentence = tokenlize(open(file_path, encoding='UTF-8').read())
            ws.fit(sentence)
    ws.build_vocab(min=10, max_features=10000)
    pickle.dump(ws, open('./model/ws.pkl', 'wb'))
    print(len(ws))
