# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on the code from Jianwei Yang
# --------------------------------------------------------
import os
import os.path
import sys
import torch.utils.data as data
import cv2
import torch
import random
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from lib.model.utils.config import cfg
import collections
import albumentations as A
from matplotlib import pyplot as plt

class MetaDataset(data.Dataset):

    """Meta Dataset
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val')
        metaclass(string): the class name
        img_size(int) : the PRN network input size
        shot(int): the number of instances
        shuffle(bool)
    """

    def __init__(self, root, image_sets, metaclass, img_size, shots=1, shuffle=False, phase=1):
        self.root = root
        self.image_set = image_sets
        self.img_size = img_size
        self.metaclass = metaclass
        self.shots = shots
        if phase == 2:
            self.shots = shots * 3
        self.shuffle=shuffle
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.shot_path  = open(os.path.join(self.root, 'VOC2007', 'ImageSets/Main/shots.txt'), 'w')  # the default saved path
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        class_to_idx = dict(zip(self.metaclass, range(len(self.metaclass))))  # class to index mapping

        self.prndata = []
        self.prncls = []
        prn_image, prn_mask = self.get_prndata()
        for i in range(shots):
            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i]))
                img = img.unsqueeze(0)
                mask = torch.from_numpy(np.array(prn_mask[key][i]))
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(3)
                imgmask = torch.cat([img, mask], dim=3)
                data.append(imgmask.permute(0, 3, 1, 2).contiguous())
                cls.append(class_to_idx[key])
            self.prncls.append(cls)
            self.prndata.append(torch.cat(data,dim=0)) 

    def __getitem__(self, index):
        return  self.prndata[index],self.prncls[index]

    def get_prndata(self):
        '''
        :return: the construct prn input data
        '''
        if self.shuffle:
            random.shuffle(self.ids)
        prn_image = collections.defaultdict(list)
        prn_mask = collections.defaultdict(list)
        classes = collections.defaultdict(int)
        for cls in self.metaclass:
            classes[cls] = 0
        for img_id in self.ids:
            target = ET.parse(self._annopath % img_id).getroot()
            img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
            img = img.astype(np.float32, copy=False)
            img -= cfg.PIXEL_MEANS
            height, width, _ = img.shape
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            h, w, _ = img.shape
            y_ration = float(h) / self.img_size
            x_ration = float(w) / self.img_size
            img_resize = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            for obj in target.iter('object'):
                difficult = int(obj.find('difficult').text) == 1
                if difficult:
                    continue
                name = obj.find('name').text.strip()
                if name not in self.metaclass:
                    continue
                if classes[name] >= self.shots:
                    break
                classes[name] += 1
                bbox = obj.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox.find(pt).text)) - 1
                    if i % 2 == 0:
                        cur_pt = int(cur_pt / x_ration)
                        bndbox.append(cur_pt)
                    elif i % 2 == 1:
                        cur_pt = int(cur_pt / y_ration)
                        bndbox.append(cur_pt)
                mask[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = 1
                prn_image[name].append(img_resize)
                prn_mask[name].append(mask)
                self.shot_path.write(str(img_id[1])+'\n')
                break
            if len(classes)>0 and min(classes.values()) == self.shots:
                break
        self.shot_path.close()
        return prn_image, prn_mask

    def __len__(self):
        return len(self.prndata)


class DiorMetaDataset(data.Dataset):

    """Meta Dataset
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val')
        metaclass(string): the class name
        img_size(int) : the PRN network input size
        shot(int): the number of instances
        shuffle(bool)
    """

    def __init__(self, root, image_sets, metaclass, img_size, shots=1, shuffle=False, phase=1):
        self.root = root
        self.image_set = image_sets
        self.img_size = img_size
        self.metaclass = metaclass
        self.shots = shots
        if phase == 2:
            self.shots = shots * 3
        self.shuffle=shuffle
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.shot_path  = open(os.path.join(self.root, 'ImageSets/Main/shots.txt'), 'w')  # the default saved path
        self.ids = list()
        for line in open(os.path.join(self.root, 'ImageSets', 'Main', self.image_set + '.txt')):
            self.ids.append((self.root, line.strip()))

        class_to_idx = dict(zip(self.metaclass, range(len(self.metaclass))))  # class to index mapping

        self.prndata = []
        self.prncls = []
        prn_image, prn_mask = self.get_prndata() 
        for i in range(shots):
            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())): 
                img = torch.from_numpy(np.array(prn_image[key][i]))
                img = img.unsqueeze(0)
                mask = torch.from_numpy(np.array(prn_mask[key][i]))
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(3)
                imgmask = torch.cat([img, mask], dim=3)
                data.append(imgmask.permute(0, 3, 1, 2).contiguous())
                cls.append(class_to_idx[key])
            self.prncls.append(cls) # list of list 
            self.prndata.append(torch.cat(data,dim=0)) 

    def __getitem__(self, index):
        return  self.prndata[index],self.prncls[index]

    def get_prndata(self):
        '''
        :return: the construct prn input data
        '''
        if self.shuffle:
            random.shuffle(self.ids)
        prn_image = collections.defaultdict(list)
        prn_mask = collections.defaultdict(list)
        classes = collections.defaultdict(int)
        for cls in self.metaclass:
            classes[cls] = 0
        for img_id in self.ids:
            target = ET.parse(self._annopath % img_id).getroot()
            img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
            img = img.astype(np.float32, copy=False)
            img -= cfg.PIXEL_MEANS
            height, width, _ = img.shape
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            h, w, _ = img.shape
            y_ration = float(h) / self.img_size
            x_ration = float(w) / self.img_size
            img_resize = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            for obj in target.iter('object'):
                name = obj.find('name').text.strip()
                if name not in self.metaclass:
                    continue
                if classes[name] >= self.shots:
                    break
                classes[name] += 1
                bbox = obj.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox.find(pt).text)) - 1
                    if i % 2 == 0:
                        cur_pt = int(cur_pt / x_ration)
                        bndbox.append(cur_pt)
                    elif i % 2 == 1:
                        cur_pt = int(cur_pt / y_ration)
                        bndbox.append(cur_pt)
                mask[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = 1
                prn_image[name].append(img_resize)
                prn_mask[name].append(mask)
                self.shot_path.write(str(img_id[1])+'\n')
                break
            if len(classes)>0 and min(classes.values()) == self.shots:
                break
        self.shot_path.close()
        return prn_image, prn_mask

    def __len__(self):
        return len(self.prndata)


class ImagePatchDiorMetaDataset(data.Dataset):

    """Meta Dataset
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val')
        metaclass(string): the class name
        img_size(int) : the PRN network input size
        shot(int): the number of instances
        shuffle(bool)
    """

    def __init__(self, root, image_sets, metaclass, img_size, shots=1, shuffle=False, phase=1):
        self.root = root
        self.image_set = image_sets
        self.img_size = img_size
        self.metaclass = metaclass
        self.shots = shots
        if phase == 2:
            self.shots = shots * 3
        self.shuffle=shuffle
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.shot_path  = open(os.path.join(self.root, 'ImageSets/Main/shots.txt'), 'w')  # the default saved path
        self.ids = list()
        for line in open(os.path.join(self.root, 'ImageSets', 'Main', self.image_set + '.txt')):
            self.ids.append((self.root, line.strip()))

        class_to_idx = dict(zip(self.metaclass, range(len(self.metaclass))))  # class to index mapping

        self.rotate_transform1 = A.Rotate(limit=180, p=1)
        self.rotate_transform2 = A.Rotate(limit=180, p=1)
        self.rotate_transform3 = A.Rotate(limit=180, p=1)
        self.rotate_transform4 = A.Rotate(limit=180, p=1)
        self.rotate_transform5 = A.Rotate(limit=180, p=1)


        self.prndata = []
        self.prncls = []
        prn_image = self.get_prndata()
        for i in range(shots):
            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i]))                 #img = torch.from_numpy(np.array(prn_image[key][i * 6]))
                img = img.unsqueeze(0)
                data.append(img.permute(0, 3, 1, 2).contiguous())
                cls.append(class_to_idx[key])
            self.prncls.append(cls)
            self.prndata.append(torch.cat(data,dim=0)) 

            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i * 6 + 1]))
                img = img.unsqueeze(0)
                data.append(img.permute(0, 3, 1, 2).contiguous())
                cls.append(class_to_idx[key])
            self.prncls.append(cls)
            self.prndata.append(torch.cat(data,dim=0))

            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i * 6 + 2]))
                img = img.unsqueeze(0)
                data.append(img.permute(0, 3, 1, 2).contiguous())
                cls.append(class_to_idx[key])
            self.prncls.append(cls)
            self.prndata.append(torch.cat(data,dim=0))

            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i * 6 + 3]))
                img = img.unsqueeze(0)
                data.append(img.permute(0, 3, 1, 2).contiguous())
                cls.append(class_to_idx[key])
            self.prncls.append(cls)
            self.prndata.append(torch.cat(data,dim=0))

            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i * 6 + 4]))
                img = img.unsqueeze(0)
                data.append(img.permute(0, 3, 1, 2).contiguous())
                cls.append(class_to_idx[key])
            self.prncls.append(cls)
            self.prndata.append(torch.cat(data,dim=0))

            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i * 6 + 5]))
                img = img.unsqueeze(0)
                data.append(img.permute(0, 3, 1, 2).contiguous())
                cls.append(class_to_idx[key])
            self.prncls.append(cls)
            self.prndata.append(torch.cat(data,dim=0))


    def __getitem__(self, index):
        return  self.prndata[index],self.prncls[index]

    def get_prndata(self):
        '''
        :return: the construct prn input data
        '''
        if self.shuffle:
            random.shuffle(self.ids)
        prn_image = collections.defaultdict(list)
        classes = collections.defaultdict(int)
        for cls in self.metaclass:
            classes[cls] = 0
        for img_id in self.ids:
            target = ET.parse(self._annopath % img_id).getroot()
            img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
            img = img[:, :, ::-1]
            img = img.astype(np.float32, copy=False)
            img -= cfg.PIXEL_MEANS
            for obj in target.iter('object'):
                name = obj.find('name').text.strip()
                if name not in self.metaclass:
                    continue
                if classes[name] >= self.shots:
                    break
                classes[name] += 1
                bbox = obj.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox.find(pt).text)) - 1
                    if i % 2 == 0:
                        cur_pt = int(cur_pt)
                        bndbox.append(cur_pt)
                    elif i % 2 == 1:
                        cur_pt = int(cur_pt)
                        bndbox.append(cur_pt)
                image_patch = crop(img, bndbox, size=self.img_size)

                rotated_image_patch1 = self.rotate_transform1(image = image_patch)['image']
                rotated_image_patch2 = self.rotate_transform2(image = image_patch)['image']
                rotated_image_patch3 = self.rotate_transform3(image = image_patch)['image']
                rotated_image_patch4 = self.rotate_transform4(image = image_patch)['image']
                rotated_image_patch5 = self.rotate_transform5(image = image_patch)['image']

                prn_image[name].append(image_patch)
                prn_image[name].append(rotated_image_patch1)
                prn_image[name].append(rotated_image_patch2)
                prn_image[name].append(rotated_image_patch3)
                prn_image[name].append(rotated_image_patch4)
                prn_image[name].append(rotated_image_patch5)

                self.shot_path.write(str(img_id[1])+'\n')
                break
            if len(classes)>0 and min(classes.values()) == self.shots:
                break
        self.shot_path.close()
        return prn_image

    def __len__(self):
        return len(self.prndata)


def crop(image, purpose, size):

    cut_image = image[int(purpose[1]):int(purpose[3]),int(purpose[0]):int(purpose[2]),:]
    height, width = cut_image.shape[0:2]
    max_hw   = max(height, width)
    cty, ctx = [height // 2, width // 2]
    cropped_image  = np.zeros((max_hw, max_hw, 3), dtype=cut_image.dtype)

    x0, x1 = max(0, ctx - max_hw // 2), min(ctx + max_hw // 2, width)
    y0, y1 = max(0, cty - max_hw // 2), min(cty + max_hw // 2, height)

    assert x0 == 0
    assert y0 == 0

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = max_hw // 2, max_hw // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = cut_image[y0:y1, x0:x1, :]

    return cv2.resize(cropped_image, dsize=(size,size), interpolation=cv2.INTER_LINEAR)
