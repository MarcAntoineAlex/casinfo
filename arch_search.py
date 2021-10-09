import os
import sys
import time
import glob
import numpy as np
import torch
import util
import logging
import argparse
import torch.nn as nn
import torch.utils
import utils
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from models.model import Informer
from util.metrics import metric
from torch.utils.data import DataLoader
from architect1 import Architect
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from util.tools import EarlyStopping, adjust_learning_rate


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--name', required=True)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=12, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=0.1, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for arch encoding')
parser.add_argument('--lambda_par', type=float, default=1.0, help='unlabeled ratio')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='/home/LAB/gaoch/asdf/data/ETDataset/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=8, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training',
                    default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# other settings
args = parser.parse_args()

args.path = os.path.join('run/search/', os.environ["SLURM_JOBID"])
args.save = args.path
try:
    os.makedirs(args.path)
except FileExistsError:
    pass
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.devices
    device = torch.device('cuda:{}'.format(args.gpu))

    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    for i in range(args.itr):
        teacher = Informer(args.enc_in, args.dec_in, args.c_out, args.seq_len, args.label_len, args.pred_len, args.factor,
                         args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.dropout, args.attn,
                         args.embed, args.freq, args.activation, args.output_attention, args.distil, args.mix,
                         device, args).float().cuda()
        assistant = Informer(args.enc_in, args.dec_in, args.c_out, args.seq_len, args.label_len, args.pred_len, args.factor,
                           args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.dropout, args.attn,
                           args.embed, args.freq, args.activation, args.output_attention, args.distil, args.mix,
                           device, args).float().cuda()
        student = Informer(args.enc_in, args.dec_in, args.c_out, args.seq_len, args.label_len, args.pred_len, args.factor,
                           args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.dropout, args.attn,
                           args.embed, args.freq, args.activation, args.output_attention, args.distil, args.mix,
                           device, args).float().cuda()
        criterion_t = nn.MSELoss().cuda()
        criterion_a = nn.MSELoss().cuda()
        criterion_s = nn.MSELoss().cuda()
        cus_loss = nn.MSELoss().cuda()

        optimizer_t = torch.optim.Adam(teacher.W(),args.learning_rate)
        optimizer_a = torch.optim.Adam(assistant.W(),args.learning_rate)
        optimizer_s = torch.optim.Adam(student.W(),args.learning_rate)

        trn_data, trn_loader = _get_data(flag='train')
        val_data, val_loader = _get_data(flag='val')
        unl_data, unl_loader = _get_data(flag='train')
        test_data, test_loader = _get_data(flag='test')

        architect = Architect(teacher, assistant, student, args, device)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        STAT_arch = []
        STAT_arch_grad = []
        STAT_arch_std = []

        for epoch in range(args.epochs):
            logging.info('epoch %d', epoch)

            # training
            train(trn_loader, val_loader, unl_loader, test_loader, teacher, assistant, student, architect,
                  criterion_t, criterion_a, criterion_s, cus_loss,
                  optimizer_t, optimizer_a, optimizer_s, args.learning_rate, epoch, early_stopping, i, STAT_arch, STAT_arch_grad, STAT_arch_std)  # todo: learning_rate ->lr

            # validation
            test(teacher)
            test(assistant)
            test(student)
            adjust_learning_rate(optimizer_t, epoch + 1, args)
            adjust_learning_rate(optimizer_a, epoch + 1, args)
            adjust_learning_rate(optimizer_s, epoch + 1, args)
            if early_stopping.early_stop:
                print("Early_stopping")
                break

        plt.figure()
        plt.subplot(221)
        plt.plot(STAT_arch)
        plt.subplot(222)
        plt.plot(STAT_arch_grad)
        plt.subplot(223)
        plt.plot(STAT_arch_std)
        plt.savefig(args.path + '/' + 'arch{}.jpg'.format(i))

        best_model_path = args.path + '/' + 'checkpoint{}.pth'.format(i)
        teacher.load_state_dict(torch.load(best_model_path))
        test(teacher)


def train(trn_loader, val_loader, unl_loader, test_loader, teacher, assistant, student, architect,
          criterion_t, criterion_a, criterion_s, cus_loss, optimizer_t, optimizer_a, optimizer_s, lr, epoch, early_stopping, i,
          STAT_arch, STAT_arch_grad, STAT_arch_std):
    loss_counter = utils.AvgrageMeter()
    data_count = 0
    for step, trn_data in enumerate(trn_loader):
        teacher.train()

        # get a random minibatch from the search queue with replacement
        try:
            val_data = next(val_iter)
        except:
            val_iter = iter(val_loader)
            val_data = next(val_iter)
    
        # get a random minibatch from the unlabeled queue with replacement
        try:
            unl_data = next(unl_iter)
        except:
            unl_iter = iter(unl_loader)
            unl_data = next(unl_iter)

        implicit_grads = architect.step_all3(trn_data, val_data, unl_data, lr, optimizer_t, optimizer_a, optimizer_s, args.unrolled, data_count)

        STAT_arch_grad.append(implicit_grads[0].mean().item())
        STAT_arch.append(teacher.architect_param123.mean().item())
        STAT_arch_std.append(teacher.architect_param123.std().item())

        optimizer_t.zero_grad()
        logit_t, true = _process_one_batch(trn_data, teacher)
        loss_t = critere(criterion_t, teacher, logit_t, true, data_count)
        loss_t.backward()
        optimizer_t.step()
    
        ##########################################################################################################
    
        optimizer_a.zero_grad()
        logit_t, _ = _process_one_batch(unl_data, teacher)
        logit_a, _ = _process_one_batch(unl_data, assistant)
        loss_a1 = cus_loss(logit_a, logit_t)

        logit_a, true = _process_one_batch(trn_data, assistant)
        loss_a2 = criterion_a(logit_a, true)

        loss_a = loss_a1 + args.lambda_par * loss_a2
        loss_a.backward()
        optimizer_a.step()

        ##########################################################################################################

        optimizer_s.zero_grad()
        logit_a, true = _process_one_batch(unl_data, assistant)
        logit_s, true = _process_one_batch(unl_data, student)
        loss_s1 = cus_loss(logit_s, logit_a.detach())

        logit_s, true = _process_one_batch(trn_data, student)
        loss_s2 = criterion_s(logit_s, true)

        loss_s = loss_s1 + args.lambda_par * loss_s2
        loss_s.backward()
        optimizer_s.step()

        ##########################################################################################################

        if step % args.report_freq == 0:
            logging.info("\tstep: {}, epoch: {} | loss: {:.7f}".format(step, epoch, loss_t.item()))
            loss_counter.update(loss_t.item())
        data_count += args.batch_size

    vali_loss = vali(val_loader, criterion_t, teacher)
    vali_loss_a = vali(val_loader, criterion_a, assistant)
    vali_loss_s = vali(val_loader, criterion_s, student)
    test_loss = vali(test_loader, criterion_t, teacher)

    logging.info("Epoch: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f} Assis_val Loss: {:.7f} Stud_val Loss: {:.7f}".format(
        epoch, loss_counter.avg, vali_loss, test_loss, vali_loss_a, vali_loss_s))
    early_stopping(vali_loss, teacher, args.path, i)


def test(teacher):
    test_data, test_loader = _get_data(flag='test')
    teacher.eval()

    preds = []
    trues = []

    for i, test_d in enumerate(test_loader):
        pred, true = _process_one_batch(test_d, teacher)
        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
    trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))

    # result save
    folder_path = './results/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    logging.info('mse:{}, mae:{}'.format(mse, mae))
    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)
    return


def vali(vali_loader, criterion, model):
    model.eval()
    total_loss = []
    for i, val_d in enumerate(vali_loader):
        pred, true = _process_one_batch(val_d, model)
        loss = criterion(pred.detach().cpu(), true.detach().cpu())
        total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def critere(criterion, teacher, pred, true, data_count, reduction='mean'):
    reweighting = torch.softmax(teacher.architect_param123[data_count:data_count + pred.shape[0]], dim=0)**0.5
    if reduction != 'mean':
        crit = nn.MSELoss(reduction=reduction)
        return crit(pred * reweighting, true * reweighting).mean(dim=-1)
    return criterion(pred * reweighting, true * reweighting)


def _get_data(flag):
    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'WTH': Dataset_Custom,
        'ECL': Dataset_Custom,
        'Solar': Dataset_Custom,
        'custom': Dataset_Custom,
    }
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.detail_freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        timeenc=timeenc,
        freq=freq,
        cols=args.cols
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


def _process_one_batch(data, model):
    batch_x = data[0].float().cuda()
    batch_y = data[1].float().cuda()

    batch_x_mark = data[2].float().cuda()
    batch_y_mark = data[3].float().cuda()

    # decoder input
    if args.padding == 0:
        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().cuda()
    elif args.padding == 1:
        dec_inp = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float().cuda()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
    # encoder - decoder
    if args.use_amp:
        with torch.cuda.amp.autocast():
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    else:
        if args.output_attention:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if args.features == 'MS' else 0
    batch_y = batch_y[:, -args.pred_len:, f_dim:].cuda()

    return outputs, batch_y


if __name__ == '__main__':
    main()
