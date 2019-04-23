import torch
from torch import nn
import argparse
import yaml
import models
from data.dataset import TextDataset, word_to_idx
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

def main ():
    with open(args.config) as f:
        config = yaml.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    model = models.__dict__[args.model](ch_size=args.ch_size, embed_dim=args.embed_dim, vocab_size=len(word_to_idx))
    if args.gpu:
        model = model.cuda()
    if args.dataparallel:
        model = nn.DataParallel(model)

    train_data = TextDataset("data/train")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_data = TextDataset("data/val")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

def train(model, loader, criterion, optimizer, writer):
    for i, (input, label) in enumerate(loader):
        output = model(input)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            writer.add_scalar("Train/Loss", loss)
            writer.add_scalar("Train/Acc", torch.sum(torch.max(output, 1)[1]==label))

if __name__ == "__main__":
    main ()
