import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import os
from .model import resnet
from PIL import ImageFilter
import random

def adjust_learning_rate(args, optimizer, epoch, lr):
    # if args.cosine:
    #     eta_min = lr * (args.lr_decay_rate ** 3)
    #     lr = eta_min + (lr - eta_min) * (
    #             1 + math.cos(math.pi * epoch / args.epochs)) / 2
    # else:
    steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
    if steps > 0:
        lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def ERG(arr, k=2):
    arr_clone = arr.copy()
    arr_sum = arr_clone.sum(axis=0)
    index = np.argsort(arr_sum)[-k:]
    for i in index:
        arr_clone[i] = 0
        arr_clone[:, i] = 0
    arr_sum = arr_clone.sum(axis=0)
    return arr_sum.argmax(), index

def get_tsne_feature(model, loader, device, opt):
    model.eval()
    with torch.no_grad():
        for idx, (image, label, room_name, image_name) in enumerate(loader):
            if torch.cuda.is_available():
                image = image.to(device)
                label = label.to(device)
            if opt.mode == 'label':
                feat, _ = model(image)
            elif opt.mode == 'self':
                feat, _ = model(image)
            feat = feat.detach().cpu().numpy()
            label = label.cpu().numpy()
            room_name = np.array(room_name)
            image_name = np.array(image_name)
            sim = np.matmul(feat, feat.T)
            np.fill_diagonal(sim, 0)
            index = sim.sum(axis=0).argmax()
            new_index, ex = ERG(sim, 3)
            print(image_name[index], image_name[new_index])
            print(image_name[ex])
            if idx == 0:
                tsne_data = feat
                tsne_label = label
                tsne_room_name = room_name
                tsne_image_name = image_name
            else:
                tsne_data = np.concatenate((tsne_data, feat), axis=0)
                tsne_label = np.concatenate((tsne_label, label), axis=0)
                tsne_room_name = np.concatenate((tsne_room_name, room_name), axis=0)
                tsne_image_name = np.concatenate((tsne_image_name, image_name), axis=0)
    return tsne_data, tsne_label, tsne_room_name, tsne_image_name

def plot_tsne(data, label, room_name, image_name):
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = 2
    color = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    tsne = TSNE(n_components=2, random_state=0)
    data = tsne.fit_transform(data)
    # room = ['FloorPlan26', 'FloorPlan227', 'FloorPlan328', 'FloorPlan429', 'FloorPlan30']
    kitchen_room = ['FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30']
    living_room = ['FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230']
    bed_room = ['FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330']
    bath_room = ['FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430']
    # room = []
    # room.append(random.sample(kitchen_room, 1))
    # room.append(random.sample(living_room, 1))
    # room.append(random.sample(bed_room, 1))
    # room.append(random.sample(bath_room, 1))
    total_room = ['FloorPlan26', 'FloorPlan27', 'FloorPlan28', 'FloorPlan29', 'FloorPlan30', 'FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230', 'FloorPlan326', 'FloorPlan327', 'FloorPlan328', 'FloorPlan329', 'FloorPlan330', 'FloorPlan426', 'FloorPlan427', 'FloorPlan428', 'FloorPlan429', 'FloorPlan430']
    room = random.sample(total_room, 2)
    plt.figure(figsize=(12, 10))
    for i in range(2):
        if i == 0:
            output_data = data[room_name==room[i]]
            output_label = label[room_name==room[i]]
            output_image_name = image_name[room_name==room[i]]
            output_room_name = room_name[room_name==room[i]]
        else:
            output_data = np.concatenate((output_data, data[room_name==room[i]]), axis=0)
            output_label = np.concatenate((output_label, label[room_name==room[i]]), axis=0)
            output_image_name = np.concatenate((output_image_name, image_name[room_name==room[i]]), axis=0)
            output_room_name = np.concatenate((output_room_name, room_name[room_name==room[i]]), axis=0)
        plt.scatter(data[room_name==room[i], 0], data[room_name==room[i], 1], marker='.', label=room[i], c=color[i])
    plt.legend()
    plt.show()
    return output_data, output_image_name, output_label, output_room_name

def save_model(model, optimizer, epoch, model_path):
    print('==> Saving...')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_out_path = "best_model.pth"
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    model_out_path = os.path.join(model_path, model_out_path)
    torch.save(state, model_out_path)

def set_model(opt, device):
    if opt.mode == 'label':
        model = resnet.resnet18(num_classes=4, opt=opt)
        # model = resnet.resnet50(num_classes=4, opt=opt)
    elif opt.mode == 'self':
        model = resnet.resnet50(opt=opt)
    if opt.pretrained == '':
        # pass
        if opt.mode == 'label':
            # pass
            pretrained_model = torchvision.models.resnet18(pretrained=True)
            # pretrained_model = torchvision.models.resnet50(pretrained=True)
            model.conv1.load_state_dict(pretrained_model.conv1.state_dict())
            model.bn1.load_state_dict(pretrained_model.bn1.state_dict())
            model.relu.load_state_dict(pretrained_model.relu.state_dict())
            model.maxpool.load_state_dict(pretrained_model.maxpool.state_dict())
            model.layer1.load_state_dict(pretrained_model.layer1.state_dict())
            model.layer2.load_state_dict(pretrained_model.layer2.state_dict())
            model.layer3.load_state_dict(pretrained_model.layer3.state_dict())
            model.layer4.load_state_dict(pretrained_model.layer4.state_dict())
            model.avgpool.load_state_dict(pretrained_model.avgpool.state_dict())
        else:
            pass
    else:
        checkpoint = torch.load(opt.pretrained)
        model.load_state_dict(checkpoint['model'], strict=False)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count(), 'Multi GPU running')
            model = torch.nn.DataParallel(model)
        model = model.to(device)

    return model


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x