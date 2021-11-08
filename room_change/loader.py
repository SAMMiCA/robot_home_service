from torchvision.transforms import (ColorJitter, Compose, Normalize,
                                    RandomCrop, RandomHorizontalFlip, Resize,
                                    Scale, ToTensor)
import torch.utils.data as data
import torch
import os, random
from PIL import Image
class Random_two_room_loader(data.Dataset):
    def __init__(self, root='/data/klleon/workspace/active_learning_rearrange-main/data_collection', mode='test'):
        super(Random_two_room_loader, self).__init__()
        self.root = os.path.join(root, mode)

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = Compose(
            [
                Resize(224),
                ToTensor(),
                normalize
            ]
        )
        self.room = []
        if mode == 'train':
            for i in range(20):
                self.room.append('FloorPlan' + str(i + 1))
                self.room.append('FloorPlan' + str(i + 201))
                self.room.append('FloorPlan' + str(i + 301))
                self.room.append('FloorPlan' + str(i + 401))
        elif mode == 'val':
            for i in range(20, 25):
                self.room.append('FloorPlan' + str(i + 1))
                self.room.append('FloorPlan' + str(i + 201))
                self.room.append('FloorPlan' + str(i + 301))
                self.room.append('FloorPlan' + str(i + 401))
        elif mode == 'test':
            for i in range(25, 30):
                self.room.append('FloorPlan' + str(i + 1))
                self.room.append('FloorPlan' + str(i + 201))
                self.room.append('FloorPlan' + str(i + 301))
                self.room.append('FloorPlan' + str(i + 401))
        self.combination()

    def make_combination(self):
        self.combination = []
        for i in range(100):
            a = random.sample(self.room, 1)
            b = random.sample(self.room, 1)
            self.combination.append([a, b])

    def __len__(self):
        return len(self.combination)

    def __getitem__(self, item):
        before = self.combination[item][0]
        after = self.combination[item][1]

        before_path = os.path.join(self.root, before)
        after_path = os.path.join(self.root, after)

class Room_loader(data.Dataset):
    def __init__(self, root='/data/lab/workspace/active_learning_rearrange-main/data_collection3', mode='train'):
        super(Room_loader, self).__init__()
        self.root = root
        self.mode = mode
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if mode == 'train':
            self.transform = Compose(
                [
                    Resize((256, 256)),
                    RandomHorizontalFlip(0.5),
                    RandomCrop(224, padding=0),
                    # Resize(224),
                    # RandomHorizontalFlip(),
                    # ColorJitter(0.4, 0.4, 0.4, 0.1),
                    ToTensor(),
                    normalize
                ]
            )
        else:
            self.transform = Compose(
                [
                    Resize(224),
                    ToTensor(),
                    normalize
                ]
            )

        image_file = os.path.join(root, mode + '_images.txt')
        class_file = os.path.join(root, mode + '_image_class_label.txt')

        id2image = self.list2dict(self.text_read(image_file))
        id2class = self.list2dict(self.text_read(class_file))

        self.images = []
        self.labels = []
        for k in id2image.keys():
            image_path = os.path.join(root, id2image[k])
            self.images.append(image_path)
            self.labels.append(int(id2class[k]))

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item]
        name = self.images[item]
        img = Image.open(name).convert('RGB')
        img = self.transform(img)
        # for i in range(8):
        #     img = Image.open(name.replace(f'{0:03d}', f'{i:03d}')).convert('RGB')
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     if i == 0:
        #         img_concat = img
        #     else:
        #         img_concat = torch.cat((img_concat, img), dim=0)
        if self.mode == 'train':
            return img, label
        else:
            room_name = self.images[item].split('/')[7]
            image_name = self.images[item].split('/')[-1]
            return img, label, room_name, image_name

class Self_Room_loader(data.Dataset):
    def __init__(self, root='/data/klleon/workspace/active_learning_rearrange-main/data_collection', mode='train', transform=None):
        super(Self_Room_loader, self).__init__()
        self.root = root
        self.mode = mode
        self.transform = Compose(transform)

        image_file = os.path.join(root, mode + '_images.txt')
        class_file = os.path.join(root, mode + '_image_class_label.txt')

        id2image = self.list2dict(self.text_read(image_file))
        id2class = self.list2dict(self.text_read(class_file))

        self.images = []
        self.labels = []
        for k in id2image.keys():
            image_path = os.path.join(root, id2image[k])
            self.images.append(image_path)
            self.labels.append(int(id2class[k]))

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        q = self.transform(img)
        k = self.transform(img)

        return q, k

