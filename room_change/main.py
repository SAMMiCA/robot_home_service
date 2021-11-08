import os, argparse, time, sys, random

from PIL import Image
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import (ColorJitter, Compose, Normalize,
                                    RandomCrop, RandomHorizontalFlip, Resize,
                                    Scale, ToTensor)

from .model import resnet
from .loader import Room_loader, Self_Room_loader
from .utils import adjust_learning_rate, AverageMeter, accuracy, get_tsne_feature, plot_tsne, save_model, set_model, GaussianBlur


def parse_args():
    parser = argparse.ArgumentParser('argument for training')
    ### Training setting ###
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')

    ### optimization ###
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default=[10,30],
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # parser.add_argument('--pretrained', type=str, default='')
    # parser.add_argument('--pretrained', type=str, default='./save/models/202110152005-label/best_model.pth')
    parser.add_argument('--pretrained', type=str, default='./save/models/202110221651-label/best_model.pth')
    # parser.add_argument('--pretrained', type=str, default='./save/models/202110042117-scale30/best_model.pth')
    parser.add_argument('--tsne', action='store_true', default=False)
    parser.add_argument('--nolog', action='store_true', default=False)
    parser.add_argument('--change_test', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='label', choices=['self', 'label'])
    opt = parser.parse_args()
    if opt.change_test:
        opt.nolog = True
    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'
    model_name = time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '-' + opt.mode

    opt.save_path = os.path.join(opt.model_path, model_name)
    if not os.path.isdir(opt.save_path):
        if not opt.nolog:
            os.makedirs(opt.save_path)

    opt.tb_path = os.path.join(opt.tb_path, model_name)
    if not os.path.isdir(opt.tb_path):
        if not opt.nolog:
            os.makedirs(opt.tb_path)


    return opt

def train(model, criterion, optimizer, epoch, loader):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    CEloss = AverageMeter()
    end = time.time()
    for idx, (image, label) in enumerate(loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            image = image.to(device)
            label = label.to(device)
        bsz = label.shape[0]
        feat, logit = model(image)
        CE_loss = criterion(logit, label)
        optimizer.zero_grad()
        CE_loss.backward()
        optimizer.step()
        CEloss.update(CE_loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'CE loss {CEloss.val:.3f} ({CEloss.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, CEloss=CEloss))
            sys.stdout.flush()

    return CEloss.avg

def self_train(model, criterion, optimizer, epoch, loader):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for idx, (image1, image2) in enumerate(loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            image1 = image1.to(device)
            image2 = image2.to(device)
        bsz = image1.shape[0]

        p1, z1 = model(image1)
        p2, z2 = model(image2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 5 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {CEloss.val:.3f} ({CEloss.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, CEloss=losses))
            sys.stdout.flush()

    return losses.avg

def self_test(model, loader, criterion):
    model.eval()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for idx, (image1, image2) in enumerate(loader):
            if torch.cuda.is_available():
                image1 = image1.to(device)
                image2 = image2.to(device)
            bsz = image1.shape[0]

            p1, z1 = model(image1)
            p2, z2 = model(image2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            loss_meter.update(loss.item(), bsz)
    return loss_meter.avg

def test(model, loader, criterion):
    model.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for idx, (image, label, _, _) in enumerate(loader):
            if torch.cuda.is_available():
                image = image.to(device)
                label = label.to(device)
            bsz = image.shape[0]
            feat, logit = model(image)
            loss = criterion(logit, label)
            acc = accuracy(logit, label)
            acc_meter.update(acc[0].item(), bsz)
            loss_meter.update(loss.item(), bsz)
    return acc_meter.avg, loss_meter.avg

def change_detection_test(model):
    model.eval()
    root = '/data/lab/workspace/active_learning_rearrange-main/data_collection4/test'
    room = []
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose(
        [
            Resize(224),
            ToTensor(),
            normalize
        ]
    )
    for i in range(25, 30):
        room.append('FloorPlan' + str(i + 1))
        room.append('FloorPlan' + str(i + 201))
        room.append('FloorPlan' + str(i + 301))
        room.append('FloorPlan' + str(i + 401))

    correct = 0
    total = 5000
    cls_correct = 0
    cls_total = 0
    combination = make_combination(room, total)
    for tup in combination:
        before = tup[0]
        before_img = None
        before_name = []
        before_path = os.path.join(root, before)
        if random.random() > 0.5:
            before_point = 1
        else:
            before_point = 0
        l = sorted(os.listdir(before_path))
        for i in range(before_point * 8, (before_point + 1) * 8):
            name = os.path.join(before_path, l[i])
            img = Image.open(name).convert('RGB')
            img = transform(img)
            if before_img is None:
                before_img = img.unsqueeze(0)
            else:
                before_img = torch.cat([before_img, img.unsqueeze(0)], dim=0)
            before_name.append(name.split('/')[-1])

        after = tup[1]
        after_img = None
        after_name = []
        after_path = os.path.join(root, after)
        if random.random() > 0.5:
            after_point = 1
        else:
            after_point = 0
        l = sorted(os.listdir(after_path))
        for i in range(after_point * 8, (after_point + 1) * 8):
            name = os.path.join(after_path, l[i])
            img = Image.open(name).convert('RGB')
            img = transform(img)
            if after_img is None:
                after_img = img.unsqueeze(0)
            else:
                after_img = torch.cat([after_img, img.unsqueeze(0)], dim=0)
            after_name.append(name.split('/')[-1])
        _, before_logit = model(before_img.to(device))
        _, after_logit = model(after_img.to(device))
        before_label = before_logit.sum(dim=0).argmax().cpu().item()
        after_label = after_logit.sum(dim=0).argmax().cpu().item()
        if before_label == after_label:
            change = 1
        else:
            change = 0
        if change == tup[2]:
            correct += 1
        if tup[2] == 0:
            cls_total += 1
            if after_label == tup[4]:
                cls_correct += 1
            # else:
            #     import pdb
            #     pdb.set_trace()
    print(correct * 100 / total)
    print(cls_correct * 100 / cls_total, cls_total, cls_correct)

def make_combination(room, total):
    combination = []
    for i in range(total):
        a = random.sample(room, 1)[0]
        b = random.sample(room, 1)[0]
        aaa = int(a[9:]) // 100
        bbb = int(b[9:]) // 100
        if aaa == 0:
            a_label = 0
        elif aaa == 2:
            a_label = 1
        elif aaa == 3:
            a_label = 2
        else:
            a_label = 3
        if bbb == 0:
            b_label = 0
        elif bbb == 2:
            b_label = 1
        elif bbb == 3:
            b_label = 2
        else:
            b_label = 3

        if aaa == bbb:
            combination.append([a, b, 1, a_label, b_label])
        else:
            combination.append([a, b, 0, a_label, b_label])

    return combination

if __name__ == '__main__':
    global device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    opt = parse_args()

    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = set_model(opt, device)

    if torch.cuda.is_available():
        model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    writer = SummaryWriter(opt.tb_path)
    if opt.change_test:
        change_detection_test(model)
    elif opt.tsne:
        train_data = Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection4', mode='train')
        valid_data = Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection4', mode='val')
        test_data = Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection4', mode='test')
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

        data, label, room_name, image_name = get_tsne_feature(model, test_loader, device, opt)
        data, image_name, label, room_name = plot_tsne(data, label, room_name, image_name)
        bandwidth = estimate_bandwidth(data, quantile=0.4)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        plt.figure(figsize=(12, 10))
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(data[my_members, 0], data[my_members, 1], col + '.', label=k)
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.legend()
        plt.show()
        # matching_list = {'FloorPlan26': 1, 'FloorPlan227':3, 'FloorPlan328':2, 'FloorPlan429':0, 'FloorPlan30':4}
        # for i in range(data.shape[0]):
        #     if matching_list[room_name[i]] != labels[i]:
        #         print(image_name[i], data[i])
    else:
        if opt.mode == 'label':
            criterion = nn.CrossEntropyLoss()

            train_data = Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection4',
                                     mode='train')
            valid_data = Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection4',
                                     mode='val')
            test_data = Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection4',
                                    mode='test')
            train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
            valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)


            best_acc = 0
            best_epoch = 0

            for epoch in range(opt.epoch):
                adjust_learning_rate(opt, optimizer, epoch, opt.learning_rate)

                loss = train(model, criterion, optimizer, epoch, train_loader)
                val_acc, val_loss = test(model, valid_loader, criterion)
                test_acc, test_loss = test(model, test_loader, criterion)
                print(' * Train Loss {:.3f}  Val Acc@1 {:.3f} Val Loss {:.3f} Test Acc@1 {:.3f} Test Loss {:.3f}'.format(loss, val_acc, val_loss, test_acc, test_loss))
                if not opt.nolog:
                    writer.add_scalar('Train/Loss', loss, epoch)
                    writer.add_scalar('Val/Acc', val_acc, epoch)
                    writer.add_scalar('Val/Loss', val_loss, epoch)
                    writer.add_scalar('Test/Acc', test_acc, epoch)
                    writer.add_scalar('Test/Loss', test_loss, epoch)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    if not opt.nolog:
                        save_model(model, optimizer, epoch, opt.save_path)

        elif opt.mode == 'self':
            criterion = nn.CosineSimilarity(dim=1).to(device)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]

            train_data = Self_Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection3', mode='train', transform=augmentation)
            valid_data = Self_Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection', mode='val', transform=augmentation)
            test_data = Self_Room_loader(root='/data/lab/workspace/active_learning_rearrange-main/data_collection', mode='test', transform=augmentation)
            train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
            valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
            best_loss = 10000000
            best_epoch = 0

            for epoch in range(opt.epoch):
                adjust_learning_rate(opt, optimizer, epoch, opt.learning_rate)

                loss = self_train(model, criterion, optimizer, epoch, train_loader)
                val_loss = self_test(model, valid_loader, criterion)
                test_loss = self_test(model, test_loader, criterion)
                print(
                    ' * Train Loss {:.3f} Val Loss {:.3f} Test Loss {:.3f}'.format(loss, val_loss, test_loss))
                if not opt.nolog:
                    writer.add_scalar('Train/Loss', loss, epoch)
                    writer.add_scalar('Val/Loss', val_loss, epoch)
                    writer.add_scalar('Test/Loss', test_loss, epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    if not opt.nolog:
                        save_model(model, optimizer, epoch, opt.save_path)
