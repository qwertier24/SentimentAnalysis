import torch
from torch.utils.data import DataLoader
import re
import jieba
import os
import pickle
import torch
import torch.nn.functional as F

word_to_idx = {"": 0}
max_len = 700


class TextDataset(torch.utils.data.Dataset):

    @classmethod
    def regex_change(cls, reviews):
        sub_regex = [
            #url
            re.compile(r"""
            (https?://)?
            ([a-zA-Z0-9]+)
            (\.[a-zA-Z0-9]+)
            (\.[a-zA-Z0-9]+)*
            (/[a-zA-Z0-9]+)*
            """, re.VERBOSE|re.IGNORECASE),

            # 日期
            re.compile("""
            年 |
            月 |
            日 |
            (周一) |
            (周二) |
            (周三) |
            (周四) |
            (周五) |
            (周六)
            """, re.VERBOSE),

            # 数字
            re.compile(r"[^a-zA-Z]\d+"),

            # 空格
            re.compile(r"\s+")
            ]

        for i in range(len(reviews)):
            for regex in sub_regex:
                reviews[i] = regex.sub(r"", reviews[i])
            assert len(reviews[i]) > 0

    @classmethod
    def preprocess(cls, reviews):
        stopwords = set()
        with open("stopwords_hit.txt") as f:
            for stopword in f:
                stopwords.add(stopword.replace('\n', ''))

        cls.regex_change(reviews)
        for i in range(len(reviews)):
            reviews[i] = jieba.lcut(reviews[i], cut_all=False)
            reviews[i] = [word for word in reviews[i] if word not in stopwords]
            for j, word in enumerate(reviews[i]):
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
                    # print(word, len(word_to_idx))
                reviews[i][j] = word_to_idx[word]
            reviews[i] = torch.Tensor(reviews[i])
            # print(i, reviews[i])
            # if reviews[i].shape[0] == 0:
            #     print(i, reviews[i])
            # assert reviews[i].shape[0] > 0

        return [r for r in reviews if len(r) > 0]

    @classmethod
    def get_reviews(cls, xml_path):
        reviews = []
        with open(os.path.join(xml_path)) as f:
            in_review = False
            cur_review = ""
            for line in f:
                if line[:7] == "<review":
                    in_review = True
                elif line[:8] == "</review":
                    reviews.append(cur_review)
                    in_review = False
                    cur_review = ""
                else:
                    cur_review += line
        return reviews

    def __init__(self, data_path):
        self.reviews = []
        posi = self.preprocess(self.get_reviews(os.path.join(data_path, "positive.txt")))
        nega = self.preprocess(self.get_reviews(os.path.join(data_path, "negative.txt")))
        self.reviews = [(F.pad(r, (0, max_len-r.shape[0]), "constant", 0), 0) for r in posi] \
            + [(F.pad(r, (0, max_len-r.shape[0]), "constant", 0), 1) for r in nega]

    def __getitem__(self, idx):
        return self.reviews[idx]

    def __len__(self):
        return len(self.reviews)


try:
    word_to_idx = pickle.load(open("word_to_idx.dump", "rb"))
except:
    "regenerating word_to_idx ..."
    total_reviews = TextDataset.preprocess(TextDataset.get_reviews("positive.txt")) \
        + TextDataset.preprocess(TextDataset.get_reviews("negative.txt")) \
        + TextDataset.preprocess(TextDataset.get_reviews("test.txt"))
    pickle.dump(word_to_idx, open("word_to_idx.dump", "wb"))

if __name__ == "__main__":
    dataset = TextDataset("train")
    loader = DataLoader(dataset=dataset,
                        batch_size=2,
                        num_workers=2,
                        shuffle=True)
    cnt = 1
    for idx, data in loader:
        cnt += 1
        print(idx, data)
    # total_reviews = TextDataset.preprocess(TextDataset.get_reviews("positive.txt")) \
    #     + TextDataset.preprocess(TextDataset.get_reviews("negative.txt")) \
    #     + TextDataset.preprocess(TextDataset.get_reviews("test.txt"))
    # pickle.dump(word_to_idx, open("word_to_idx.dump", "wb"))
    # words_to_idx = pickle.load(open("word_to_idx.dump", "rb"))
    pass
