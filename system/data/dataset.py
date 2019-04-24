import torch
from torch.utils.data import DataLoader
import re
import jieba
import os
import pickle
import torch
import torch.nn.functional as F
import random

word_to_idx = {"": 0}
max_len = 1024


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
        with open("data/stopwords_hit.txt") as f:
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
            reviews[i] = torch.LongTensor(reviews[i])
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
        def getpad(x):
            if x%2 == 0:
                return (x//2, x//2)
            else:
                return (x//2+1, x//2)
        self.reviews = []
        posi = self.preprocess(self.get_reviews(os.path.join(data_path, "positive.txt")))
        nega = self.preprocess(self.get_reviews(os.path.join(data_path, "negative.txt")))
        self.reviews = [(F.pad(r, getpad(max_len-r.shape[0]), "constant", 0), 0) for r in posi] \
            + [(F.pad(r, getpad(max_len-r.shape[0]), "constant", 0), 1) for r in nega]

    def __getitem__(self, idx):
        return self.reviews[idx]

    def __len__(self):
        return len(self.reviews)


try:
    word_to_idx = pickle.load(open("data/word_to_idx.dump", "rb"))
except:
    "regenerating word_to_idx ..."
    total_reviews = TextDataset.preprocess(TextDataset.get_reviews("data/positive.txt")) \
        + TextDataset.preprocess(TextDataset.get_reviews("data/negative.txt")) \
        + TextDataset.preprocess(TextDataset.get_reviews("data/test.txt"))
    pickle.dump(word_to_idx, open("data/word_to_idx.dump", "wb"))

def split_train_val(xml_path, num1, path1, path2):
    reviews = TextDataset.get_reviews(xml_path)
    random.shuffle(reviews)
    with open(path1, "w") as f1:
        for r in reviews[:num1]:
            f1.write("<review>\n")
            f1.write(r)
            f1.write("\n</review>\n")
    with open(path2, "w") as f2:
        for r in reviews[num1:]:
            f2.write("<review>\n")
            f2.write(r)
            f2.write("\n</review>\n")

if __name__ == "__main__":
    train_data = TextDataset("train")
    train_loader = DataLoader(dataset=train_data,
                        batch_size=2,
                        num_workers=2,
                        shuffle=True)
    val_data = TextDataset("val")
    val_loader = DataLoader(dataset=val_data,
                        batch_size=2,
                        num_workers=2,
                        shuffle=True)
    print(len(val_loader), len(train_loader))
    # split_train_val("positive.txt", 1000, "val/positive.txt", "train/positive.txt")
    # split_train_val("negative.txt", 1000, "val/negative.txt", "train/negative.txt")
    # cnt = 1
    # for idx, data in loader:
    #     cnt += 1
    #     if cnt < 10:
    #         print(idx, data)
    # total_reviews = TextDataset.preprocess(TextDataset.get_reviews("positive.txt")) \
    #     + TextDataset.preprocess(TextDataset.get_reviews("negative.txt")) \
    #     + TextDataset.preprocess(TextDataset.get_reviews("test.txt"))
    # pickle.dump(word_to_idx, open("word_to_idx.dump", "wb"))
    # words_to_idx = pickle.load(open("word_to_idx.dump", "rb"))
    pass
