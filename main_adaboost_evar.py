import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data_evar import EVARDataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchnet
from torchnet.meter import AUCMeter
import copy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--ensamble_size', default=8, type=int, metavar='N',
                    help='number of models in ensamble')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--fine_tune',  type=bool, default=True,
                    help='dataset name or folder')

def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    from models.evar_model import WeightedCrossEntropyLoss
    criterion = WeightedCrossEntropyLoss()
    criterion.type(args.type)
    model.type(args.type)

    #val_transform=transforms.Compose()
    val_data = EVARDataset(train_val='val') #, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    #train_transform = transforms.Compose()
    train_data = EVARDataset(train_val='train')#, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    train_loader_no_shuffle = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)
    ensambel_dict = {}
    ww = torch.ones(train_data.__len__()).cuda(args.gpus[0])/train_data.__len__()
    alpha_vec = torch.zeros(args.ensamble_size).cuda(args.gpus[0])
    val_top1_all=torch.zeros(args.ensamble_size)
    val_auc_all=torch.zeros(args.ensamble_size)
    for mm in range(args.ensamble_size):
        model.init_model()
        for epoch in range(args.start_epoch, args.epochs):
            optimizer = adjust_optimizer(optimizer, epoch, regime)

            # train for one epoch
            train_loss,train_top1,auc_train,_= train(
                train_loader, model, criterion,ww, epoch, optimizer)
            train_loss,train_top1,auc_val,Iym= validate(
                train_loader_no_shuffle, model, criterion, ww,epoch)
                # evaluate on validation set
            val_loss,val_top1,auc_val,_ = validate(
                val_loader, model, criterion,ww, epoch)
            logging.info('\n Ensamble Model: {0}\t'
                         'Epoch: {1}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Acc {train_top1:.4f} \t'
                         'Validation Loss {val_loss:.4f}\t'
                         'Validation Acc {val_top1:.4f}\n'
                         .format(mm + 1,epoch + 1, train_loss=train_loss, train_top1=train_top1,val_loss=val_loss,val_top1=val_top1))

            results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss)
            results.save()
        val_top1_all[mm]=val_top1
        val_auc_all[mm]=auc_val[0]
        err=ww.dot(Iym)/ww.sum()
        alpha=torch.log((1-err)/err)
        ww=ww.mul(torch.exp(alpha.mul(Iym)))
        alpha_vec[mm]=alpha
        ensambel_dict[str(mm)] = copy.deepcopy(model)

    ensamble_val_top1,ensamble_auc_val = validate_ensamble(val_loader, model, criterion, epoch,alpha_vec, ensambel_dict)
    logging.info('\n Ensamble Acc After Epoch: {0}\t'
                 'Validation AUC {ensamble_auc_val:.4f}\t'
                 'Validation TOP1 {ensamble_val_top1:.4f}\n'
                 .format(epoch + 1, ensamble_auc_val=ensamble_auc_val[0],ensamble_val_top1=ensamble_val_top1))
    print('Val AUC: ',val_auc_all.mean(),' Val Top-1: ',val_top1_all.mean())
        # remember best prec@1 and save checkpoint
        #is_best = val_prec1 > best_prec1
        #best_prec1 = max(val_prec1, best_prec1)

    if val_top1>70.71:
        import pdb; pdb.set_trace()
    save_checkpoint({
        'epoch': epoch + 1,
        'model': args.model,
        'config': args.model_config,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'regime': regime
    }, False, path=save_path)

def forward(data_loader, model, criterion,weight_index=None, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    auc_meter=AUCMeter()
    end = time.time()
    Iym=None

    for i, (input,target,idx) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.float().cuda(async=True)
        input_var   = Variable(input.type(args.type), volatile=not training)
        #target_var     = Variable(target.long())
        idx=idx.type(args.type).long()
        #import pdb; pdb.set_trace()
        # compute output

        output = model(input_var)
        #import pdb; pdb.set_trace()
        t_onehot = torch.FloatTensor(target.size(0), 2).type(args.type)
        t_onehot.zero_()
        import pdb; pdb.set_trace()
        t_onehot.scatter_(1, target.long().unsqueeze(1), 1)
        target_var     = Variable(t_onehot)

        loss = criterion(output, target_var,idx,weight_index)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        _,pred=output.data.max(1)
        Iym=pred.float().ne(target).float() if Iym is None else torch.cat((Iym,pred.float().ne(target).float()),0)
        prec1= accuracy(output.data, target.long())
        losses.update(loss.data[0], input.size(0))
        #import pdb; pdb.set_trace()
        top1.update(prec1[0][0], input.size(0))
        auc_meter.add(output.data.max(1)[1],target)
        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses,top1=top1))
            #print('accuracy',top1.avg)

    return losses.avg, top1.avg,auc_meter.value(),Iym

def validate_ensamble(data_loader, model, criterion, epoch=0, alpha_vec=None, ensambel_dict=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    auc_meter=AUCMeter()
    end = time.time()

    for i, (input,target,idx) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.float().cuda(async=True)
        input_var   = Variable(input.type(args.type), volatile=True)
        #target_var     = Variable(target.long())
        adaboost_pred = None

        for key in ensambel_dict:
            model = ensambel_dict[key]
            output = model(input_var)
            t_onehot = torch.FloatTensor(target.size(0), 2).type(args.type)
            t_onehot.zero_()
            t_onehot.scatter_(1, target.long().unsqueeze(1), 1)
            target_var     = Variable(t_onehot)
            loss = criterion(output, target_var,idx,weights=None)
            if type(output) is list:
                output = output[0]
        # measure accuracy and record loss
            _,pred=output.data.max(1)
            adaboost_pred=pred.float().mul(2).add(-1).mul(alpha_vec[int(key)]) if adaboost_pred is None else adaboost_pred.add(pred.float().mul(2).add(-1).mul(alpha_vec[int(key)]))

        adaboost_pred=adaboost_pred.sign()
        y_onehot = torch.FloatTensor(adaboost_pred.size(0), 2)
        y_onehot.zero_()
        y_onehot.scatter_(1, adaboost_pred.add(1).div(2).long().cpu().unsqueeze(1), 1)
        prec1 = accuracy(y_onehot.cuda(), target.long())
        top1.update(prec1[0][0], input.size(0))
        auc_meter.add(adaboost_pred.add(1).div(2).long(),target)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % int(args.print_freq/10) == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='EVALUATING ENSAMBLE',
                             batch_time=batch_time,
                             data_time=data_time,top1=top1))
            #print('accuracy',top1.avg)
    return top1.avg,auc_meter.value()
def train(data_loader, model, criterion,weight_index, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, weight_index, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, weight_index,epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, weight_index, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
