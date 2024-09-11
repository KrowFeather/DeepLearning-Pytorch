class Word2Sequece:
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}
        self.inverse_dict = {}

    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)
        for word in self.count:
            self.dict[word] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]

        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        return [self.inverse_dict.get(idx) for idx in indices]

    def __len__(self):
        return len(self.dict)

# ws = Word2Sequece()
# ws.fit(['我', '是', '谁'])
# ws.fit(['我', '是', '我'])
# ws.build_vocab(min=0)
# print(ws.dict)
# ret = ws.transform(['我', '爱', '北京'], max_len=10)
# print(ret)
# ret = ws.inverse_transform(ret)
# print(ret)
