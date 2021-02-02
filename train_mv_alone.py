"""Run training."""

import shutil
from tarfile import filemode
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.nn.modules import module
import torch.nn.parallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset_mga_model import CoviarDataSet
from mga_model import IframeNet, MvNet
from train_options import parser
import torchvision
from transforms import *
from config import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score
SAVE_FREQ = 5
PRINT_FREQ = 20
ACCUMU_STEPS = 1  # use gradient accumlation to use least memory and more runtime
loss_min = 1
CONTINUE_FROM_LAST = False
LAST_SAVE_PATH = r'kinetics400_resnet34_bt_120_seg_3_fixed_gop11_checkpoint.pth.tar'
FINETUNE = True

# for visualization
filename=0
WRITER = []
DEVICES = []
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="2,3"
description = ""
cfg = Config()
cfg.parse({'train_data_root': r'/home/sjhu/datasets/UCF-101-mpeg4',
            'test_data_root': r'/home/sjhu/datasets/UCF-101-mpeg4',
           'dataset': 'ucf101',
            'model': 'resnet34',
           'train_list': r'/home/sjhu/datasets/ucf101_split1_train.txt',
           'test_list' : r'/home/sjhu/datasets/ucf101_split1_test.txt',
           'gpus': [0,1],
           'batch_size': 120,
           'alpha':4,
           'num_segments':3,
           'workers':32,
           'lr': 0.01,
           'lr_steps': [50,100,150],
           'epochs': 510
           })

# cfg.parse({'train_data_root': r'/home/sjhu/datasets/kinetics400_mpeg4/train_256',
#            'test_data_root': r'/home/sjhu/datasets/kinetics400_mpeg4/val_256',
#            'dataset': 'kinetics400',
#            'num_class': 400,
#            'model': 'resnet34',
#            'train_list': r'/home/sjhu/datasets/k400_train_with_numframes.txt',
#            'test_list': r'/home/sjhu/datasets/k400_val_with_numframes.txt',
#            # 'lr': (1.6*cfg.batch_size*dist.get_world_size())/1024,
#            'lr': 0.02,
#            'batch_size': 120,
#            'alpha': 4,
#            'num_segments': 3,
#            'workers': 48,
#            'weight_decay': 1e-4,
#            'eval_freq': 5,
#            'lr_steps': [50,90,120],
#            'epochs': 300,
#            'gpus': [0,1]
#            })

def main():
    print(torch.cuda.device_count())
    global devices
    global WRITER

    global description
    description = '%s_%s_bt_%d_seg_%d_%s' % (cfg.dataset,cfg.model,cfg.batch_size*ACCUMU_STEPS, cfg.num_segments,'fixed_gop11_sgd')
    log_name = './log/%s' % description
    WRITER = SummaryWriter(log_name)
    print('Training arguments:')
    for k, v in vars(cfg).items():
        print('\t{}: {}'.format(k, v))

    # model = IframeNet(cfg.num_segments)
    model = MvNet(cfg.num_segments,cfg.num_class,"resnet34")
    if FINETUNE:
        checkpoint = torch.load(LAST_SAVE_PATH)
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        pretrained_dict = {k: v for k, v in base_dict.items() if 'fc' not in k}
        model.load_state_dict(pretrained_dict,strict=False)
        print("Load pretrained model..... Finished.")

    # add continue train from before
    if CONTINUE_FROM_LAST:
        checkpoint = torch.load(LAST_SAVE_PATH)
        # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        print("model epoch {} lowest loss {}".format(checkpoint['epoch'], checkpoint['loss_min']))
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        loss_min = checkpoint['loss_min']
        model.load_state_dict(base_dict)
        start_epochs = checkpoint['epoch']
    else:
        loss_min = 10000
        start_epochs = 0

    # print(model)
    # WRITER.add_graph(model, (torch.randn(10,5, 2, 224, 224),))

    devices = [torch.device("cuda:%d" % device) for device in cfg.gpus]
    global DEVICES
    DEVICES = devices

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            cfg.train_data_root,
            cfg.dataset,
            video_list=cfg.train_list,
            num_segments=cfg.num_segments,
            iframe_transform = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(224, [1, .875, .75, .66]),
             GroupRandomHorizontalFlip(is_mv=False)]),
            mv_transform = torchvision.transforms.Compose(
            [GroupMultiScaleCrop(224, [1, .875, .75]),
             GroupRandomHorizontalFlip(is_mv=True)]),
            is_train=True,
            accumulate=True,
        ),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            cfg.test_data_root,
            cfg.dataset,
            video_list=cfg.test_list,
            num_segments=cfg.num_segments,
            iframe_transform = torchvision.transforms.Compose([
                GroupScale(256),
                GroupCenterCrop(224),
                ]),
            mv_transform= torchvision.transforms.Compose([
                GroupScale(256),
                GroupCenterCrop(224),
                ]),
            is_train=False,
            accumulate=True,
        ),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=cfg.gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key or 'bn' in key else 1.0
        if 'conv1' in key or 'data_bn' in key:
            lr_mult = 0.1
        elif '.fc.' in key:
            lr_mult = 1.0
        elif 'conv1x1' in key:
            lr_mult = 0.1
        else:
            lr_mult = 0.01
            

        params += [{'params': value, 'lr': cfg.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]


    if FINETUNE:
        optimizer = torch.optim.SGD(params,momentum=0.9,weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(
        params,
        weight_decay=cfg.weight_decay,
        eps=0.001)

    criterions = torch.nn.CrossEntropyLoss().to(devices[0])
    # criterions = LabelSmoothingLoss(101,0.1,-1)

    # try to use ReduceOnPlatue to adjust lr
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20 // cfg.eval_freq, verbose=True)

    for epoch in range(start_epochs, cfg.epochs):
        # about optimizer
        cur_lr = adjust_learning_rate(optimizer, epoch, cfg.lr_steps, cfg.lr_decay)
        WRITER.add_scalar('Lr/epoch', get_lr(optimizer), epoch)
        loss_train, acc_train = train(train_loader, model, criterions, optimizer, epoch)
        if epoch % cfg.eval_freq == 0 or epoch == cfg.epochs - 1:
            loss_val, acc_val = validate(val_loader, model, criterions, epoch)
            # scheduler.step(loss_val)
            is_best = (loss_val < loss_min)
            loss_min = min(loss_val, loss_min)
            # visualization
            WRITER.add_scalars('Accuracy/epoch', {'Train': acc_train, 'Val': acc_val}, epoch)
            WRITER.add_scalars('Classification Loss/epoch', {'Train': loss_train, 'Val': loss_val}, epoch)
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'loss_min': loss_min,
                    },
                    is_best,
                    filename='checkpoint.pth.tar')
    WRITER.close()


def train(train_loader, model, criterions, optimizer, epoch):
    '''
    :param train_loader:
    :param model:
    :param criterions:
    :param optimizer:
    :param epoch:
    :return:  (clf loss)
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    clf_losses = AverageMeter()

    model.train()
    end = time.time()
    correct_nums=0
    for i, (input_pairs, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_pairs[0] = input_pairs[0].float().to(devices[0])
        input_pairs[1] = input_pairs[1].float().to(devices[0])
        label = label.float().to(devices[0])
        fconv1,flayer1,flayer2,flayer3,flayer4,fv,outputs = model(input_pairs[1])
        loss = criterions(outputs, label.clone().long()) / ACCUMU_STEPS
        clf_losses.update(loss.item(), cfg.batch_size)
        loss.backward()

        # use gradient accumulation
        if i % ACCUMU_STEPS == 0:
            # attention the following line can't be transplaced
            optimizer.step()
            optimizer.zero_grad()

        _, predicts = torch.max(outputs, 1)
        correct_nums += (predicts == label.clone().long()).sum()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}],\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'classifier Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'.format(
                epoch, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss2=clf_losses)))

    acc = float(100 * correct_nums) / len(train_loader.dataset)
    return clf_losses.avg ,acc  # attention indent ,there was a serious bug here


def validate(val_loader, model, criterions, epoch):
    '''
    :param val_loader:
    :param model:
    :param criterions:
    :param epoch:
    :return:  (clf loss, acc)
    '''
    batch_time = AverageMeter()
    clf_losses = AverageMeter()
    model.eval()
    end = time.time()
    correct_nums = 0

    scores = []
    labels = []

    for i, (input_pairs, label) in enumerate(val_loader):
        with torch.no_grad():
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            label = label.float().to(devices[0])

            _,_,_,_,_,_,y = model(input_pairs[1])
            loss = criterions(y, label.clone().long()) / ACCUMU_STEPS
            clf_losses.update(loss.item(), cfg.batch_size)

            _, predicts = torch.max(y, 1)
            correct_nums += (predicts == label.clone().long()).sum()
            scores.append(y.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print(('Validate: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'clf loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                    .format(
                    i, len(val_loader),
                    batch_time=batch_time,
                    loss2=clf_losses)))

    acc = float(100 * correct_nums) / len(val_loader.dataset)
    print((
        'Validating Results: classification loss {loss3.avg:.5f}, Accuracy: {accuracy:.3f}%'.format(loss3=clf_losses,
                                                                                                    accuracy=acc)))
    return clf_losses.avg, acc


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((description, filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((description, '_best.pth.tar'))
        shutil.copyfile(filename, best_name)


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(cfg.lr_steps)))
    lr = cfg.lr * decay
    wd = cfg.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def visualizeFM(feature_map):
    global filename
    filename = filename +1
    import torchvision.utils as vutil
    fm = feature_map.clone().detach()
    fm = fm.permute(1,0,2,3)
    vutil.save_image(fm,filename,pad_value=0.5)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == '__main__':
    main()
