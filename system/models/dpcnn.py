import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Block(nn.Module):
    def __init__(self, ch_size, downsample=None):
        super(Block, self).__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm1d(ch_size)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(ch_size, ch_size, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(ch_size)
        self.conv2 = nn.Conv1d(ch_size, ch_size, 5, padding=2)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual


class DPCNN(nn.Module):
    def __init__(self, ch_size, embed_dim, vocab_size, max_len, mark_size):
        super(DPCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed2 = nn.Embedding(mark_size, embed_dim, padding_idx=0)
        self.mark_size = mark_size

        self.region_embed = nn.Sequential(nn.Conv1d(embed_dim*2, ch_size, 5, padding=2),
                                          nn.Dropout(0.2))

        x_len = max_len
        blocks = []
        blocks.append(Block(ch_size))
        downsample = nn.Sequential(nn.ConstantPad1d(padding=(0, 1), value=0),
                                   nn.MaxPool1d(3, stride=2))
        while x_len > 1:
            blocks.append(Block(ch_size, downsample))
            x_len //= 2

        self.blocks = nn.Sequential(*blocks)

        self.linear = nn.Linear(x_len * ch_size, 2)
        self.dropout = nn.Dropout(0.5)

    def init_embeds(self, pretrained_embeds, train_embed, train_embed2):
        self.embed.weight.data.copy_(torch.from_numpy(np.array(pretrained_embeds)))
        # embed2_weight = torch.zeros(self.embed2.num_embeddings, self.embed2.embedding_dim)
        # for i in range(1, self.mark_size):
        #     embed2_weight[i,i] = 1
        # self.embed2.weight.data.copy_(embed2_weight)
        self.embed.requires_grad = train_embed
        self.embed2.requires_grad = train_embed2


    def forward(self, x):
        N = x.shape[0]

        # Region embedding
        x0 = self.embed(x[:,0,:].squeeze(1))
        x1 = self.embed(x[:,1,:].squeeze(1))
        x = torch.cat((x0, x1), 2).permute(0, 2, 1).contiguous()

        x = self.region_embed(x)

        x = self.blocks(x)

        x = x.view(N, -1)
        x = self.dropout(x)
        x = self.linear(x)

        return x
