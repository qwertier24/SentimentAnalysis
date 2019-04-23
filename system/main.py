import torch
from torch import nn
import argparse
import yaml
import models


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

def main ():
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config.items():
        setattr(args, k, v)

    model = models.__dict__ [args.model](ch_size=args.ch_size, embed_dim=args.embed_dim)
    embeds = nn.Embedding (word_num, )
    if args.gpu:
        model = model.cuda()
    if args.dataparallel:
        model = nn.DataParallel(model)

if __name__ == "__main__":
    main ()
