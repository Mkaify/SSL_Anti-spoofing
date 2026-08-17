import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed

__author__ = "Hemlata Tak | MSS-Urdu adapter: Kaif / MSP Lab"


def evaluate_accuracy(dev_loader, model, device):
    val_loss  = 0.0
    num_total = 0.0
    model.eval()
    weight    = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in dev_loader:
        batch_size  = batch_x.size(0)
        num_total  += batch_size
        batch_x     = batch_x.to(device)
        batch_y     = batch_y.view(-1).type(torch.int64).to(device)
        batch_out   = model(batch_x)
        batch_loss  = criterion(batch_out, batch_y)
        val_loss   += (batch_loss.item() * batch_size)
    val_loss /= num_total
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    for batch_x, utt_id in data_loader:
        fname_list  = []
        score_list  = []
        batch_x     = batch_x.to(device)
        batch_out   = model(batch_x)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
    print('Scores saved to {}'.format(save_path))


def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_total    = 0.0
    model.train()
    weight    = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in train_loader:
        batch_size   = batch_x.size(0)
        num_total   += batch_size
        batch_x      = batch_x.to(device)
        batch_y      = batch_y.view(-1).type(torch.int64).to(device)
        batch_out    = model(batch_x)
        batch_loss   = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    running_loss /= num_total
    return running_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SSL Anti-Spoofing — MSS-Urdu')

    # ── Paths ──────────────────────────────────────────────────────────────
    parser.add_argument('--database_path', type=str,
        default='/kaggle/input/mss-urdu-dataset/MSS/ULA/MSS_urdu',
        help='Root of MSS_urdu folder containing MSS_urdu_train/flac, '
             'MSS_urdu_dev/flac, MSS_urdu_eval/flac')

    parser.add_argument('--protocols_path', type=str,
        default='/kaggle/input/mss-urdu-protocols/MSS_urdu_cm_protocols',
        help='Folder containing cm.train.trn.txt, cm.dev.trl.txt, cm.eval.trl.txt')

    # ── Hyperparameters ─────────────────────────────────────────────────────
    parser.add_argument('--batch_size',   type=int,   default=14)
    parser.add_argument('--num_epochs',   type=int,   default=100)
    parser.add_argument('--lr',           type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss',         type=str,   default='weighted_CCE')

    # ── Model ───────────────────────────────────────────────────────────────
    parser.add_argument('--seed',       type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--comment',    type=str, default=None)

    # ── Auxiliary ───────────────────────────────────────────────────────────
    parser.add_argument('--track',       type=str, default='LA',
                        choices=['LA', 'PA', 'DF'])
    parser.add_argument('--eval_output', type=str, default=None)
    parser.add_argument('--eval',        action='store_true', default=False)
    parser.add_argument('--is_eval',     action='store_true', default=False)
    parser.add_argument('--eval_part',   type=int, default=0)

    # ── Backend ─────────────────────────────────────────────────────────────
    parser.add_argument('--cudnn-deterministic-toggle',
                        action='store_false', default=True)
    parser.add_argument('--cudnn-benchmark-toggle',
                        action='store_true',  default=False)

    # ── RawBoost ────────────────────────────────────────────────────────────
    parser.add_argument('--algo', type=int, default=5)
    parser.add_argument('--nBands',           type=int, default=5)
    parser.add_argument('--minF',             type=int, default=20)
    parser.add_argument('--maxF',             type=int, default=8000)
    parser.add_argument('--minBW',            type=int, default=100)
    parser.add_argument('--maxBW',            type=int, default=1000)
    parser.add_argument('--minCoeff',         type=int, default=10)
    parser.add_argument('--maxCoeff',         type=int, default=100)
    parser.add_argument('--minG',             type=int, default=0)
    parser.add_argument('--maxG',             type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f',              type=int, default=5)
    parser.add_argument('--P',                type=int, default=10)
    parser.add_argument('--g_sd',             type=int, default=2)
    parser.add_argument('--SNRmin',           type=int, default=10)
    parser.add_argument('--SNRmax',           type=int, default=40)

    # ────────────────────────────────────────────────────────────────────────

    if not os.path.exists('models'):
        os.mkdir('models')

    args = parser.parse_args()
    set_random_seed(args.seed, args)

    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        args.track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model     = Model(args, device)
    nb_params = sum([p.view(-1).size()[0] for p in model.parameters()])
    model     = model.to(device)
    print('nb_params: {}'.format(nb_params))

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Model loaded: {}'.format(args.model_path))

    # ── Eval ────────────────────────────────────────────────────────────────
    if args.eval:
        eval_proto = os.path.join(args.protocols_path, 'cm.eval.trl.txt')
        file_eval  = genSpoof_list(dir_meta=eval_proto, is_train=False, is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(
            list_IDs=file_eval,
            base_dir=os.path.join(args.database_path, 'MSS_urdu_eval', 'flac'))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    # ── Train ────────────────────────────────────────────────────────────────
    train_proto = os.path.join(args.protocols_path, 'cm.train.trn.txt')
    d_label_trn, file_train = genSpoof_list(dir_meta=train_proto,
                                            is_train=True, is_eval=False)
    print('no. of training trials', len(file_train))
    train_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_train,
        labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'MSS_urdu_train', 'flac'),
        algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=2, shuffle=True, drop_last=True)
    del train_set, d_label_trn

    # ── Dev ──────────────────────────────────────────────────────────────────
    dev_proto = os.path.join(args.protocols_path, 'cm.dev.trl.txt')
    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_proto,
                                          is_train=False, is_eval=False)
    print('no. of validation trials', len(file_dev))
    dev_set = Dataset_ASVspoof2019_train(
        args,
        list_IDs=file_dev,
        labels=d_label_dev,
        base_dir=os.path.join(args.database_path, 'MSS_urdu_dev', 'flac'),
        algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,
                            num_workers=2, shuffle=False)
    del dev_set, d_label_dev

    # ── Training loop ────────────────────────────────────────────────────────
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(args.num_epochs):
        running_loss = train_epoch(train_loader, model, args.lr, optimizer, device)
        val_loss     = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('val_loss', val_loss,     epoch)
        writer.add_scalar('loss',     running_loss, epoch)
        print('\nepoch {} | train_loss {:.4f} | val_loss {:.4f}'.format(
              epoch, running_loss, val_loss))
        torch.save(model.state_dict(),
                   os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
