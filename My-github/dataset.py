import os
from PIL import Image
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import random
import cv2
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed

class CoData_Train(data.Dataset):
    def __init__(self, img_root, gt_root, sal_root, img_size, transform, max_num, jigsaw=True):
        # Path Pool
        self.img_root = img_root  # root
        self.gt_root = gt_root
        self.sal_root = sal_root


        self.dirs = os.listdir(img_root)  # all dir
        self.img_dir_paths = list(  # [img_root+dir1, ..., img_root+dir2]
            map(lambda x: os.path.join(img_root, x), self.dirs))
        self.gt_dir_paths = list(  # [gt_root+dir1, ..., gt_root+dir2]
            map(lambda x: os.path.join(gt_root, x), self.dirs))
        self.sal_dir_paths = list(  # [gt_root+dir1, ..., gt_root+dir2]
            map(lambda x: os.path.join(sal_root, x), self.dirs))
        self.img_name_list = [os.listdir(idir) for idir in self.img_dir_paths
                              ]  # [[name00,..., 0N],..., [M0,..., MN]]
        self.gt_name_list = [
            map(lambda x: x[:-3] + 'png', iname_list)
            for iname_list in self.img_name_list
        ]
        self.sal_name_list = [
            map(lambda x: x[:-3] + 'png', iname_list)
            for iname_list in self.img_name_list
        ]
        self.img_path_list = [
            list(
                map(lambda x: os.path.join(self.img_dir_paths[idx], x),
                    self.img_name_list[idx]))
            for idx in range(len(self.img_dir_paths))
        ]  # [[impath00,..., 0N],..., [M0,..., MN]]
        self.gt_path_list = [
            list(
                map(lambda x: os.path.join(self.gt_dir_paths[idx], x),
                    self.gt_name_list[idx]))
            for idx in range(len(self.gt_dir_paths))
        ]  # [[gtpath00,..., 0N],..., [M0,..., MN]]

        self.sal_path_list = [
            list(
                map(lambda x: os.path.join(self.sal_dir_paths[idx], x),
                    self.sal_name_list[idx]))
            for idx in range(len(self.sal_dir_paths))
        ]  # [[gtpath00,..., 0N],..., [M0,..., MN]]
        self.nclass = len(self.dirs)

        # Other Hyperparameters
        self.size = img_size
        self.cat_size = int(img_size * 2)
        self.sizes = [img_size, img_size]
        self.transform = transform
        self.max_num = max_num
        self.jigsaw = jigsaw

    def __getitem__(self, item):
        img_paths = self.img_path_list[item]
        gt_paths = self.gt_path_list[item]
        sal_paths = self.sal_path_list[item]
        num = len(img_paths)
        # 随机sample不超过max_num的物体
        if num > self.max_num:
            sampled_list = random.sample(range(num), self.max_num)
            new_img_paths = [img_paths[i] for i in sampled_list]
            img_paths = new_img_paths
            new_gt_paths = [gt_paths[i] for i in sampled_list]
            gt_paths = new_gt_paths
            new_sal_paths = [sal_paths[i] for i in sampled_list]
            sal_paths = new_sal_paths
            num = self.max_num

        imgs = torch.Tensor(num, 3, self.sizes[0], self.sizes[1])
        gts = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])
        sals = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])
        distractors = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            gt = Image.open(gt_paths[idx]).convert('L')
            sal = Image.open(sal_paths[idx]).convert('L')
            distractor = Image.fromarray(np.uint8(np.zeros(self.sizes)), mode='L')

            subpaths.append(
                os.path.join(img_paths[idx].split('/')[-2],
                             img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((img.size[1], img.size[0]))

            # Co-Data Augmentation
            if self.jigsaw and random.random() > 0.25:
                [img, gt, sal, distractor] = self.jigsaw2(img_paths[idx],
                                                     gt_paths[idx],
                                                     sal_paths[idx],
                                                     dir_item=item)

            [img, gt, sal, distractor] = self.transform(img, gt, sal, distractor)

            imgs[idx] = img
            gts[idx] = gt
            sals[idx] = sal
            distractors[idx] = distractor

        return imgs, gts, sals, subpaths, ori_sizes, distractors

    def __len__(self):
        return len(self.dirs)

    def jigsaw2(self, img_path, gt_path, sal_path, dir_item):

        obj_img = cv2.resize(cv2.imread(img_path), (self.size, self.size),
                             interpolation=cv2.INTER_LINEAR)
        obj_gt = cv2.resize(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE),
                            (self.size, self.size),
                            interpolation=cv2.INTER_NEAREST)
        obj_sal = cv2.resize(cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE),
                            (self.size, self.size),
                            interpolation=cv2.INTER_NEAREST)

        # Sample an additional foreground objects
        Candidate_list = list(range(self.nclass))
        Candidate_list.remove(dir_item)
        addi_item = random.choice(Candidate_list)
        addi_idx = random.randint(0, len(self.img_path_list[addi_item]) - 1)
        addi_img_path = self.img_path_list[addi_item][addi_idx]
        addi_gt_path = self.gt_path_list[addi_item][addi_idx]
        addi_sal_path = self.sal_path_list[addi_item][addi_idx]

        addi_img = cv2.resize(cv2.imread(addi_img_path),
                              (self.size, self.size),
                              interpolation=cv2.INTER_LINEAR)
        addi_gt = cv2.resize(cv2.imread(addi_gt_path, cv2.IMREAD_GRAYSCALE),
                             (self.size, self.size),
                             interpolation=cv2.INTER_NEAREST)
        addi_sal = cv2.resize(cv2.imread(addi_sal_path, cv2.IMREAD_GRAYSCALE),
                             (self.size, self.size),
                             interpolation=cv2.INTER_NEAREST)

        if random.random() < 0.5:
            img = np.zeros([self.size, self.cat_size, 3])
            gt = np.zeros([self.size, self.cat_size])
            sal = np.zeros([self.size, self.cat_size])

            distractor = np.zeros([self.size, self.cat_size])

            if random.random() < 0.5:
                img[0:self.size, 0:self.size] = obj_img
                img[0:self.size, self.size:2 * self.size] = addi_img
                gt[0:self.size, 0:self.size] = obj_gt
                gt[0:self.size, self.size:2 * self.size] = addi_gt * 0
                sal[0:self.size, 0:self.size] = obj_sal
                sal[0:self.size, self.size:2 * self.size] = addi_sal
                distractor[0:self.size, 0:self.size] = obj_gt * 0
                distractor[0:self.size, self.size:2 * self.size] = addi_gt
            else:
                img[0:self.size, 0:self.size] = addi_img
                img[0:self.size, self.size:2 * self.size] = obj_img
                gt[0:self.size, 0:self.size] = addi_gt * 0
                gt[0:self.size, self.size:2 * self.size] = obj_gt
                sal[0:self.size, 0:self.size] = addi_sal
                sal[0:self.size, self.size:2 * self.size] = obj_sal
                distractor[0:self.size, 0:self.size] = addi_gt
                distractor[0:self.size, self.size:2 * self.size] = obj_gt * 0
        else:
            img = np.zeros([self.cat_size, self.size, 3])
            gt = np.zeros([self.cat_size, self.size])
            sal = np.zeros([self.cat_size, self.size])
            distractor = np.zeros([self.cat_size, self.size])

            if random.random() < 0.5:
                img[0:self.size, 0:self.size] = obj_img
                img[self.size:2 * self.size, 0:self.size] = addi_img
                gt[0:self.size, 0:self.size] = obj_gt
                gt[self.size:2 * self.size, 0:self.size] = addi_gt * 0
                sal[0:self.size, 0:self.size] = obj_sal
                sal[self.size:2 * self.size, 0:self.size] = addi_sal
                distractor[0:self.size, 0:self.size] = obj_gt * 0
                distractor[self.size:2 * self.size, 0:self.size] = addi_gt
            else:
                img[0:self.size, 0:self.size] = addi_img
                img[self.size:2 * self.size, 0:self.size] = obj_img
                gt[0:self.size, 0:self.size] = addi_gt * 0
                gt[self.size:2 * self.size, 0:self.size] = obj_gt
                sal[0:self.size, 0:self.size] = addi_sal
                sal[self.size:2 * self.size, 0:self.size] = obj_sal
                distractor[0:self.size, 0:self.size] = addi_gt
                distractor[self.size:2 * self.size, 0:self.size] = obj_gt * 0

        return Image.fromarray(cv2.cvtColor(
            np.uint8(img), cv2.COLOR_BGR2RGB)), Image.fromarray(
                np.uint8(gt), mode='L'), Image.fromarray(np.uint8(sal), mode='L'), Image.fromarray(np.uint8(distractor),
                                                         mode='L')


class CoData_Test(data.Dataset):
    def __init__(self, img_root, gt_root, sal_root, img_size):

        class_list = os.listdir(img_root)
        self.sizes = [img_size, img_size]

        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))
        self.gt_dirs = list(
            map(lambda x: os.path.join(gt_root, x), class_list))
        self.sal_dirs = list(
            map(lambda x: os.path.join(sal_root, x), class_list))


        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_sal = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),

        ])

    def __getitem__(self, item):
        names = os.listdir(self.img_dirs[item])
        gt_names = [
            iname[:-3] + 'png'
            for iname in names
        ]
        sal_names = [
            iname[:-3] + 'png'
            for iname in names
        ]
        #print(names)
        num = len(names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), names))
        #print(img_paths)
        gt_paths = list(
            map(lambda x: os.path.join(self.gt_dirs[item], x), gt_names))
        sal_paths = list(
            map(lambda x: os.path.join(self.sal_dirs[item], x), sal_names))


        imgs = torch.Tensor(num, 3, self.sizes[0], self.sizes[1])
        sals = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])
        gts = torch.Tensor(num, 1, self.sizes[0], self.sizes[1])
        subpaths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            sal = Image.open(sal_paths[idx]).convert('L')
            gt = Image.open(gt_paths[idx]).convert('L')
            subpaths.append(
                os.path.join(img_paths[idx].split('/')[-2],
                             img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((img.size[1], img.size[0]))
            img = self.transform(img)
            imgs[idx] = img
            gt = self.transform_sal(gt)
            gts[idx] = gt
            sal = self.transform_sal(sal)
            sals[idx] =sal

        return imgs,gts, sals, subpaths, ori_sizes

    def __len__(self):
        return len(self.img_dirs)


    def __len__(self):
        return len(self.img_dirs)

class FixedResize(object):
    def __init__(self, size):
        self.sizes = (size, size)  # size: (h, w)

    def __call__(self, img, gt, sal, distractor):
        # assert img.size == gt.size

        img = img.resize(self.sizes, Image.BILINEAR)
        gt = gt.resize(self.sizes, Image.NEAREST)
        sal = sal.resize(self.sizes, Image.NEAREST)
        distractor = distractor.resize(self.sizes, Image.NEAREST)

        return img, gt,sal, distractor


class ToTensor(object):
    def __call__(self, img, gt, sal, distractor):

        return F.to_tensor(img), F.to_tensor(gt), F.to_tensor(sal), F.to_tensor(distractor)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img, gt, sal, distractor):
        img = F.normalize(img, self.mean, self.std)

        return img, gt, sal, distractor


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt, sal, distractor):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            sal = sal.transpose(Image.FLIP_LEFT_RIGHT)
            distractor = distractor.transpose(Image.FLIP_LEFT_RIGHT)

        return img, gt, sal, distractor


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, gt, sal, distractor):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        img = F.rotate(img, angle, Image.BILINEAR, self.expand, self.center)
        gt = F.rotate(gt, angle, Image.NEAREST, self.expand, self.center)
        sal = F.rotate(sal, angle, Image.NEAREST, self.expand, self.center)
        distractor = F.rotate(distractor, angle, Image.NEAREST, self.expand,
                              self.center)

        return img, gt, sal, distractor


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt, sal, distractor):
        for t in self.transforms:
            img, gt, sal, distractor = t(img, gt, sal, distractor)
        return img, gt, sal, distractor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# get the dataloader (Note: without data augmentation)
def get_loader(img_root,
               gt_root,
               sal_root,
               img_size,
               batch_size,
               max_num=float('inf'),
               istrain=True,
               jigsaw=True,
               shuffle=False,
               num_workers=0,
               pin=False):
    if istrain:
        transform = Compose([
            FixedResize(img_size),
            RandomHorizontalFlip(),
            RandomRotation((-90, 90)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = CoData_Train(img_root, gt_root, sal_root, img_size, transform, max_num, jigsaw)
    else:
        dataset = CoData_Test(img_root, gt_root, sal_root, img_size)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader


'''img_root = '/home/nku/New_Co-Sal/Dataset/DUTS_class/img'
gt_root = '/home/nku/New_Co-Sal/Dataset/DUTS_class/gt'
sal_root = '/home/nku/New_Co-Sal/Dataset/DUTS_class/sal'
img_size = 256
batch_size = 1


a = get_loader(img_root, gt_root, sal_root, img_size, batch_size, max_num=20, istrain=True, jigsaw=True, shuffle=False, num_workers=4, pin=True)

for batch in a:
    imgs = batch[0]
    gts = batch[1]
    sals = batch[2]
    subpaths = batch[3]
    for num in range(len(subpaths)):
        os.makedirs('/home/nku/New_Co-Sal/gt/'+ subpaths[num][0].split('/')[0], exist_ok=True)
        os.makedirs('/home/nku/New_Co-Sal/sal/' + subpaths[num][0].split('/')[0], exist_ok=True)

        save_tensor_img(gts[:, num, :, : ,:], '/home/nku/New_Co-Sal/gt/'+ subpaths[num][0])
        save_tensor_img(sals[:, num, :, :, :], '/home/nku/New_Co-Sal/sal/' + subpaths[num][0])'''
    #print(subpaths)
    #break