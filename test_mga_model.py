"""Run training."""

import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset_mga_model import CoviarDataSet
from mga_model import IframeNet, MvNet
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score
from config import *
from transforms import *
import os
import torchvision

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3`"
description = ""
LAST_SAVE_PATH = r'ucf101_resnet101_bt_64_seg_3_mgc_layer234__best.pth.tar'
PRINT_FREQ = 10
cfg = Config()
cfg.parse({'train_data_root': r'/home/sjhu/datasets/UCF-101-mpeg4',
            'test_data_root': r'/home/sjhu/datasets/UCF-101-mpeg4',
           'dataset': 'ucf101',
            'model': 'resnet101',
           'train_list': r'/home/sjhu/datasets/ucf101_split1_train.txt',
           'test_list' : r'/home/sjhu/datasets/ucf101_split1_test.txt',
           'gpus': [0,1,2,3],
           'batch_size': 4,
           'grad_accumu_steps': 1,
           'alpha':4,
           'num_segments':25,
           'workers':32,
           'lr': 0.01,
           'lr_steps': [120,200,390],
           'epochs': 510,
           'test_crops': 1
           })

def main():
    devices = [torch.device("cuda:%d" % device) for device in cfg.gpus]
    mvnet = MvNet(cfg.num_segments*cfg.test_crops,cfg.num_class,"resnet34")
    model = IframeNet(cfg.num_segments*cfg.test_crops,cfg.num_class,mvnet,cfg.model)
    checkpoint = torch.load(LAST_SAVE_PATH)
    print("model epoch {} max top1 acc {}".format(checkpoint['epoch'], checkpoint['top1_max']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=cfg.gpus)
    model = model.to(devices[0])
    cudnn.benchmark = True
    print("load saved model success")

    # print(model)
    # WRITER.add_graph(model, (torch.randn(10,5, 2, 224, 224),))


    if cfg.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScaleFM(256),
            GroupCenterCropFM(224),
        ])
    elif cfg.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSampleFM(224, 256)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(cfg.test_crops))

    infer_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            cfg.test_data_root,
            cfg.dataset,
            video_list=cfg.test_list,
            num_segments=cfg.num_segments,
            transform = cropping,
            is_train=False,
            accumulate=True,
        ),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)
    

    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    correct_nums = 0

    scores = []
    labels = []

    for i, (input_pairs, label) in enumerate(infer_loader):
        with torch.no_grad():
            input_pairs[0] = input_pairs[0].float().to(devices[0])
            input_pairs[1] = input_pairs[1].float().to(devices[0])
            label = label.float().to(devices[0])

            y,_,_ = model(input_pairs)
            _, predicts = torch.max(y, 1)
            correct_nums += (predicts == label.clone().long()).sum()
            scores.append(y.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print(('Validate: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                    i, len(infer_loader),
                    batch_time=batch_time)))

    acc = float(100 * correct_nums) / len(infer_loader.dataset)
    print((
        'Validating Results:  Accuracy: {accuracy:.3f}%'.format(accuracy=acc)))



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




if __name__ == '__main__':
    main()
