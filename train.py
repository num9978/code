
import os
import sys
import numpy as np
import time, datetime
import glob
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

from utils.utils import *
from utils import utils
from utils import KD_loss
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable
from mbv3 import mobilenet_v3_large

parser = argparse.ArgumentParser("mbv3")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--meta_abin_epochs', type=int, default=2, help='num of training epochs')
parser.add_argument('--meta_abin_warmup_epochs', type=int, default=1, help='num of training epochs')
parser.add_argument('--abin_epochs', type=int, default=18, help='num of training epochs')
parser.add_argument('--awbin_epochs', type=int, default=120, help='num of training epochs')
parser.add_argument('--wbits', type=int, default=4, help='num bits')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay')
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--data', type=str, default='/SSD/ILSVRC2012')
parser.add_argument('--teacher', type=str, default='resnet101', help='path of ImageNet')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')                    
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
args = parser.parse_args()

CLASSES = 1000
TASKS = [3,4,8]
TARGET_GBOPS = 3.31
TARGET_LAMBDA = 4.

args.save = 'eval-metaqttrain-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '{%(asctime)s}-(%(process)d)-%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("args = %s", args)

    # load model
    model_teacher = None
    model_teacher = models.__dict__[args.teacher](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    model_student = mobilenet_v3_large(pretrained='IMAGENET1K_V2', tasks=TASKS)
    logging.info('student:')
    logging.info(model_student)

    # Setting num ops
    model_student.change_precision(a_bin=False, w_bin=False)
    model_student(torch.rand(1,3,224,224))
    logging.info(f"Setting bops...")
    model_student.get_gbops(do_print=True)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_ce = criterion_ce.cuda()
    criterion_kd = KD_loss.DistributionLoss()
    criterion_kd = criterion_kd.cuda()
    if model_teacher is None: criterion = criterion_ce
    else: criterion = criterion_kd

    # load data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        normalize])), 
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model_student = nn.DataParallel(model_student).cuda()

    weight_parameters = []
    alpha_parameters = []
    other_parameters = []
    for pname, p in model_student.named_parameters():
        if 'alpha' in pname:
            alpha_parameters.append(p)
        elif p.ndimension() == 4 or 'weights' in pname:
            weight_parameters.append(p)
        else:
            other_parameters.append(p)

    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate/5,)
    scheduler = None
    optimizer_arch = torch.optim.Adam(
            [{'params' : alpha_parameters, 'weight_decay' : 1e-3}],
            lr=args.learning_rate/5, betas=(0.5, 0.999))

    best_top1_epoch = 0
    best_top1_acc = 0
    start_epoch = 0
    
    ## Bit selection phase
    model_student.module.change_precision(a_bin=True, w_bin=False)

    for epoch in range(start_epoch, args.meta_abin_epochs):
        logging.info(f'====actbin_metaqt_epoch-({epoch}/{args.meta_abin_epochs}) lr-({scheduler.get_lr() if scheduler is not None else args.learning_rate/5})====')
        tic = time.time()
        train_obj, train_top1_acc, train_top5_acc = train_meta(epoch, args.meta_abin_epochs, args.meta_abin_warmup_epochs, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, optimizer_arch, w_bin=False)
        logging.info(f'{epoch}epoch train_acc: {train_top1_acc:.3f}')
        val_obj, val_top1_acc, val_top5_acc = validate_meta(epoch, args.meta_abin_epochs, val_loader, model_student, criterion_ce, w_bin=False)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_student.module.state_dict(),
            'best_top1_acc': val_top1_acc,
        }, os.path.join(args.save, f'model_meta_actbin_last.pth'))

        logging.info(f'{epoch}epoch valid_acc: {val_top1_acc}, best=({best_top1_epoch}-{best_top1_acc:.3f})')
        model_student.module.get_alpha()

        logging.info(f"Epoch time: {time.time()-tic}")

    training_time = (time.time() - start_t) / 3600
    logging.info('meta actbin total training time = {} hours'.format(training_time))

    # Fixing bit-width from bit-meta & bit-search training
    model_student.module.get_alpha()
    model_student.module.set_searched_bit()
    avg_bit = model_student.module.get_avgbit()
    model_gbops = model_student.module.get_gbops()
    model_student.module.get_bit()
    logging.info(f"* Average_bit={avg_bit}-bit, Computation_cost={model_gbops}BOps")
    logging.info(model_student)

    best_top1_epoch = 0
    best_top1_acc= 0
    start_epoch = 0
    
    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.abin_epochs*len(train_loader))

    ## Weight training phase
    model_student.module.change_precision(a_bin=True, w_bin=False)

    for epoch in range(start_epoch, args.abin_epochs):
        logging.info(f'==actbin_epoch-({epoch}/{args.abin_epochs}) lr-({scheduler.get_lr()})==')
        tic = time.time()
        train_obj, train_top1_acc, train_top5_acc = train(epoch, args.abin_epochs, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, w_bin=False)
        logging.info(f'{epoch}epoch train_acc: {train_top1_acc:.3f}')
        val_obj, val_top1_acc, val_top5_acc = validate(epoch, args.abin_epochs, val_loader, model_student, criterion_ce, w_bin=False)

        if val_top1_acc > best_top1_acc:
            best_top1_epoch = epoch
            best_top1_acc = val_top1_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_student.module.state_dict(),
                'best_top1_acc': val_top1_acc,
            }, os.path.join(args.save, f'model_actbin_best.pth'))

        logging.info(f'{epoch}epoch valid_acc: {val_top1_acc}, best=({best_top1_epoch}-{best_top1_acc:.3f})')

        logging.info(f"Epoch time: {time.time()-tic}")

    training_time = (time.time() - start_t) / 3600
    logging.info(f'actbin total training time = {training_time} hours')

    best_top1_epoch = 0
    best_top1_acc= 0
    start_epoch = 0
    
    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.awbin_epochs*len(train_loader))
    
    model_student.module.change_precision(a_bin=True, w_bin=True)
    model_student.module.change_wbit(args.wbits)

    for epoch in range(start_epoch, args.awbin_epochs):
        logging.info(f'==actwbin_epoch-({epoch}/{args.awbin_epochs}) lr-({scheduler.get_lr()})==')
        tic = time.time()
        train_obj, train_top1_acc, train_top5_acc = train(epoch, args.awbin_epochs, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, w_bin=True)
        logging.info(f'{epoch}epoch train_acc: {train_top1_acc:.3f}')
        val_obj, val_top1_acc, val_top5_acc = validate(epoch, args.awbin_epochs, val_loader, model_student, criterion_ce, w_bin=True)

        if val_top1_acc > best_top1_acc:
            best_top1_epoch = epoch
            best_top1_acc = val_top1_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_student.module.state_dict(),
                'best_top1_acc': val_top1_acc,
            }, os.path.join(args.save, f'model_actwbin_best.pth'))

        logging.info(f'{epoch}epoch valid_acc: {val_top1_acc}, best=({best_top1_epoch}-{best_top1_acc:.3f})')

        logging.info(f"Epoch time: {time.time()-tic}")

    training_time = (time.time() - start_t) / 3600
    logging.info(f'actwbin total training time = {training_time} hours')

def train_meta(epoch, epochs, warmup_epochs, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, optimizer_arch, w_bin=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lossess = [AverageMeter('Loss', ':.4e') for _ in TASKS]
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1s = [AverageMeter('Acc@1', ':6.2f') for _ in TASKS]
    top5 = AverageMeter('Acc@5', ':6.2f')
    top5s = [AverageMeter('Acc@5', ':6.2f') for _ in TASKS]

    losses_arch = AverageMeter('Loss', ':.4e')
    top1_arch = AverageMeter('Acc@1', ':6.2f')
    top5_arch = AverageMeter('Acc@5', ':6.2f')

    model_student.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()
        n = images.size(0)
        
        ## Bit-meta training
        loss_1step = 0
        for j, bit in enumerate(TASKS):
            if w_bin: model_student.module.change_awbit(bit)
            else: model_student.module.change_abit(bit)

            logits_student = model_student(images)
            if model_teacher:
                with torch.no_grad():
                    logits_teacher = model_teacher(images)
                loss = criterion(logits_student, logits_teacher)
            else:
                loss = criterion(logits_student, target)
            loss_1step += loss
            lossess[j].update(loss.item(), n)
            prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
            top1.update(prec1.item(), n)
            top1s[j].update(prec1.item(), n)
            top5.update(prec5.item(), n)
            top5s[j].update(prec5.item(), n)

        loss_1step = loss_1step / len(TASKS)
        losses.update(loss_1step.item(), n)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_student.parameters(), 10.)
        optimizer.step()

        ## Bit-search training
        if epoch >= warmup_epochs:
            model_student.module.change_abit(TASKS)
            logits_student = model_student(images)
            if model_teacher:
                with torch.no_grad():
                    logits_teacher = model_teacher(images)
                loss_arch = criterion(logits_student, logits_teacher)
            else:
                loss_arch = criterion(logits_student, target)
            
            if i % args.report_freq == 0 or i == len(train_loader)-1:
                model_gbops = model_student.module.get_gbops(do_print=True)
            else:
                model_gbops = model_student.module.get_gbops()
            loss_arch = loss_arch + TARGET_LAMBDA*(model_gbops - TARGET_GBOPS).abs()

            losses_arch.update(loss_arch.item(), n)
            prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
            top1_arch.update(prec1.item(), n)
            top5_arch.update(prec5.item(), n)

            optimizer_arch.zero_grad()
            loss_arch.backward()
            optimizer_arch.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.report_freq == 0 or i == len(train_loader)-1:
            logging.info(f'train: ({epoch:03d}/{epochs:03d})-({i:03d}/{len(train_loader):03d})-(objs:{losses.avg:0.3f}-{[l.avg for l in lossess]})-(acc:{top1.avg:0.3f}-{[t.avg for t in top1s]})')
            if epoch >= warmup_epochs:
                logging.info(f'       (arch_objs:{losses_arch.avg:.3f})-(arch_acc:{top1_arch.avg:.3f})-(bops:{model_gbops.data.item():.3f}gbops)')

    return losses.avg, top1.avg, top5.avg

def validate_meta(epoch, epochs, val_loader, model, criterion, w_bin=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lossess = [AverageMeter('Loss', ':.4e') for _ in TASKS]
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1s = [AverageMeter('Acc@1', ':6.2f') for _ in TASKS]
    top5 = AverageMeter('Acc@5', ':6.2f')
    top5s = [AverageMeter('Acc@5', ':6.2f') for _ in TASKS]
    
    losses_arch = AverageMeter('Loss', ':.4e')
    top1_arch = AverageMeter('Acc@1', ':6.2f')
    top5_arch = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            n = images.size(0)

            loss_1step = 0
            for j, bit in enumerate(TASKS):
                if w_bin: model.module.change_awbit(bit)
                else: model.module.change_abit(bit)

                logits = model(images)
                loss = criterion(logits, target)
                loss_1step += loss
                lossess[j].update(loss.item(), n)
                prec1, prec5 = accuracy(logits, target, topk=(1, 5))
                top1.update(prec1.item(), n)
                top1s[j].update(prec1.item(), n)
                top5.update(prec5.item(), n)
                top5s[j].update(prec5.item(), n)
            loss_1step = loss_1step / len(TASKS)
            losses.update(loss_1step.item(), n)
            
            model.module.change_abit(TASKS)
            logits = model(images)
            model_gbops = model.module.get_gbops()
            loss_arch = criterion(logits, target) + TARGET_LAMBDA*(model_gbops - TARGET_GBOPS).abs()
            losses_arch.update(loss_arch.item(), n)
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            top1_arch.update(prec1.item(), n)
            top5_arch.update(prec5.item(), n)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.report_freq == 0 or i == len(val_loader)-1:
                logging.info(f'val: ({epoch:03d}/{epochs:03d})-({i:03d}/{len(val_loader):03d})-(objs:{losses.avg:0.3f}-{[l.avg for l in lossess]})-(acc:{top1.avg:0.3f}-{[t.avg for t in top1s]})')
                logging.info(f'     (arch_objs:{losses_arch.avg:0.3f})-(arch_acc:{top1_arch.avg:0.3f})')
        
    return losses.avg, top1.avg, top5.avg

def train(epoch, epochs, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, w_bin=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model_student.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        logits_student = model_student(images)
        if model_teacher:
            with torch.no_grad():
                logits_teacher = model_teacher(images)
            loss = criterion(logits_student, logits_teacher)
        else:
            loss = criterion(logits_student, target)

        n = images.size(0)
        losses.update(loss.item(), n)
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_student.parameters(), 10.)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.report_freq == 0 or i == len(train_loader)-1:
            logging.info(f'train: ({epoch:03d}/{epochs:03d})-({i:03d}/{len(train_loader):03d})-(lr:{scheduler.get_lr()})-(objs:{losses.avg:0.3f})-(acc:{top1.avg:0.3f}))')

        scheduler.step()
    return losses.avg, top1.avg, top5.avg

def validate(epoch, epochs, val_loader, model, criterion, w_bin=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            n = images.size(0)

            logits = model(images)
            loss = criterion(logits, target)
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.report_freq == 0 or i == len(val_loader)-1:
                logging.info(f'val: ({epoch:03d}/{epochs:03d})-({i:03d}/{len(val_loader):03d})-(objs:{losses.avg:0.3f})-(acc:{top1.avg:0.3f})')
        
    return losses.avg, top1.avg, top5.avg
    
if __name__ == '__main__':
    main()
