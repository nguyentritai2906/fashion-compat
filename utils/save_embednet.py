from __future__ import print_function

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import Resnet_18
from polyvore_outfits import TripletImageLoader
from tripletnet import Tripletnet
from type_specific_network import TypeSpecificNet

# Training settings
def parse_args():
    parser = argparse.ArgumentParser(description='Fashion Compatibility Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=256,
                        metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of start epoch (default: 1)')
    parser.add_argument('--lr',
                        type=float,
                        default=5e-5,
                        metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='enables CUDA training')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=250,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name',
                        default='Type_Specific_Fashion_Compatibility',
                        type=str,
                        help='name of experiment')
    parser.add_argument(
        '--polyvore_split',
        default='nondisjoint',
        type=str,
        help=
        'specifies the split of the polyvore data (either disjoint or nondisjoint)'
    )
    parser.add_argument(
        '--datadir',
        default='data',
        type=str,
        help='directory of the polyvore outfits dataset (default: data)')
    parser.add_argument('--test',
                        dest='test',
                        action='store_true',
                        default=False,
                        help='To only run inference on test set')
    parser.add_argument('--dim_embed',
                        type=int,
                        default=64,
                        metavar='N',
                        help='how many dimensions in embedding (default: 64)')
    parser.add_argument(
        '--use_fc',
        action='store_true',
        default=False,
        help='Use a fully connected layer to learn type specific embeddings.')
    parser.add_argument('--learned',
                        dest='learned',
                        action='store_true',
                        default=False,
                        help='To learn masks from random initialization')
    parser.add_argument('--prein',
                        dest='prein',
                        action='store_true',
                        default=False,
                        help='To initialize masks to be disjoint')
    parser.add_argument(
        '--rand_typespaces',
        action='store_true',
        default=False,
        help=
        'randomly assigns comparisons to type-specific embeddings where #comparisons < #embeddings'
    )
    parser.add_argument(
        '--num_rand_embed',
        type=int,
        default=4,
        metavar='N',
        help='number of random embeddings when rand_typespaces=True')
    parser.add_argument(
        '--l2_embed',
        dest='l2_embed',
        action='store_true',
        default=False,
        help='L2 normalize the output of the type specific embeddings')
    parser.add_argument(
        '--learned_metric',
        dest='learned_metric',
        action='store_true',
        default=False,
        help='Learn a distance metric rather than euclidean distance')
    parser.add_argument('--margin',
                        type=float,
                        default=0.3,
                        metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--embed_loss',
                        type=float,
                        default=5e-4,
                        metavar='M',
                        help='parameter for loss for embedding norm')
    parser.add_argument('--mask_loss',
                        type=float,
                        default=5e-4,
                        metavar='M',
                        help='parameter for loss for mask norm')
    parser.add_argument(
        '--vse_loss',
        type=float,
        default=5e-3,
        metavar='M',
        help='parameter for loss for the visual-semantic embedding')
    parser.add_argument('--sim_t_loss',
                        type=float,
                        default=5e-5,
                        metavar='M',
                        help='parameter for loss for text-text similarity')
    parser.add_argument('--sim_i_loss',
                        type=float,
                        default=5e-5,
                        metavar='M',
                        help='parameter for loss for image-image similarity')
    parser.add_argument('--output',
                        type=str,
                        help='name of output file')
    return parser.parse_args()

def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    fn = os.path.join(args.datadir, 'polyvore_outfits',
                      'polyvore_item_metadata.json')
    meta_data = json.load(open(fn, 'r'))
    text_feature_dim = 6000
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(TripletImageLoader(
        args,
        'test',
        meta_data,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
        ])),
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              **kwargs)

    # model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
    model = torch.load('encoder.pt')
    for param in model.parameters():
        param.requires_grad = False
    csn_model = TypeSpecificNet(args, model,
                                len(test_loader.dataset.typespaces))

    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    tnet = Tripletnet(args, csn_model, text_feature_dim, criterion)

    best_acc = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, encoding='latin1')
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    torch.save(tnet.embeddingnet, args.output)
    exit()

if __name__ == "__main__":
    main(parse_args())