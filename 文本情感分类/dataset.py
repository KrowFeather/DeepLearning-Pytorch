from torch.utils.data import Dataset, DataLoader
import os
import re


def tokenlize(content):
    content = re.sub("<.*?>", " ", content)
    filters = ['\t', '\n', '\x97', '\x96', '#', '$', '%', '&', '\.', ':', '\(', '\)', '\*']
    content = re.sub("|".join(filters), ' ', content, flags=re.S)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens


class ImdbDataSet(Dataset):
    def __init__(self, train=True):
        super(ImdbDataSet, self).__init__()
        self.train_set_path = '../../dataset/aclImdb/train'
        self.test_set_path = '../../dataset/aclImdb/test'
        self.data_path = self.train_set_path if train else self.test_set_path
        temp_data_path = [self.data_path + '/pos', self.data_path + '/neg']
        self.total_data_path = []
        for path in temp_data_path:
            dir_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in dir_list if i.endswith('.txt')]
            self.total_data_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path = self.total_data_path[index]
        label_str = file_path.split("\\")[-2]
        label = 1 if label_str == 'pos' else 0
        content = open(file_path, encoding="utf-8").read()
        token = tokenlize(content)
        return token, label

    def __len__(self):
        return len(self.total_data_path)


def collate_fn(batch):
    content, label = list(zip(*batch))
    return content, label


def get_dataloader(train=True):
    imdb_dataset = ImdbDataSet(train=train)
    dataloader = DataLoader(dataset=imdb_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    return dataloader


for idx, (input, target) in enumerate(get_dataloader()):
    print(idx)
    print(input)
    print(target)
    break
