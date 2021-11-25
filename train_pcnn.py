# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Peizhen Shi, Bowei Yan, based on the code from Xiaopeng Yan
# --------------------------------------------------------
import _init_paths
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import random

from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
import torch.utils.data as Data
from lib.roi_data_layer.roidb import combined_roidb, rank_roidb_ratio, filter_class_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient
from lib.model.faster_rcnn.resnet import resnet
import pickle
from lib.datasets.metadata import  DiorMetaDataset, ImagePatchDiorMetaDataset
from collections import OrderedDict

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train Meta R-CNN network')
    # Define training data and Model
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='dior', type=str) 
    parser.add_argument('--net', dest='net',
                        help='Prototypecnn',
                        default='Prototypecnn', type=str)
    # Define display and save dir
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=31, type=int) 
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display', 
                        default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        default='models/meta/first',
                        type=str)
    # Define training parameters
    
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)    #8
    parser.add_argument('--cuda', dest='cuda', default=True, type=bool,
                        help='whether use CUDA')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=4, type=int)
    parser.add_argument('--cag', dest='class_agnostic', default=False, type=bool,
                        help='whether perform class_agnostic bbox regression')
    # Define meta parameters
    parser.add_argument('--meta_train', dest='meta_train', default=True, type=bool, 
                        help='whether perform meta training')
    parser.add_argument('--meta_loss', dest='meta_loss', default=True, type=bool,
                        help='whether perform adding meta loss')
    parser.add_argument('--phase', dest='phase',
                        help='the phase of training process',
                        default=1, type=int) 
    parser.add_argument('--shots', dest='shots',
                        help='the number meta input of PRN network',
                        default=3, type=int) 
    parser.add_argument('--meta_type', dest='meta_type', default=1, type=int, 
                        help='choose which sets of metaclass')
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=4, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    # resume trained model 
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool) # True
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=200, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model', 
                        default=20, type=int) 
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=4285, type=int) 
    # log and diaplaysssss
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)
    parser.add_argument('--log_dir', dest='log_dir',
                        help='directory to save logs', default='checkpoint',
                        type=str)
    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    if args.use_tfboard:
        writer = SummaryWriter(args.log_dir)
    args.dataset == "dior"
    if args.phase == 1:
        if args.meta_type == 1:
            args.imdb_name = "dior_train_split_11"
        elif args.meta_type == 2:
            args.imdb_name = "dior_train_split_22"
        elif args.meta_type == 3:
            args.imdb_name = "dior_train_split_33"
        elif args.meta_type == 4:
            args.imdb_name = "dior_train_split_44"
        elif args.meta_type == 5:
             args.imdb_name = "dior_train_split_55"
    else:
        args.imdb_name = "dior_shots"
    args.imdbval_name = "dior_test" 
    args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

     # the number of sets of metaclass
    cfg.TRAIN.META_TYPE = args.meta_type

    cfg.USE_GPU_NMS = args.cuda
    if args.cuda:
        cfg.CUDA = True

    args.cfg_file = "cfgs/dior.yml"
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if args.phase == 1:
        # First phase only use the base classes
        shots = 200
        if args.meta_type == 1:  #  use the first sets of base classes
            metaclass = cfg.TRAIN.BASECLASSES_FIRST
        if args.meta_type == 2:  #  use the second sets of base classes
            metaclass = cfg.TRAIN.BASECLASSES_SECOND
        if args.meta_type == 3:  #  use the third sets of base classes
            metaclass = cfg.TRAIN.BASECLASSES_THIRD
        if args.meta_type == 4:
            metaclass = cfg.TRAIN.BASECLASSES_FOURTH
        if args.meta_type == 5:
            metaclass = cfg.TRAIN.BASECLASSES_FSODM
    else:
        # Second phase only use fewshot number of base and novel classes
        shots = args.shots
        if args.meta_type == 1:  #  use the first sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_FIRST
        if args.meta_type == 2:  #  use the second sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_SECOND
        if args.meta_type == 3:  #  use the third sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_THIRD
        if args.meta_type == 4:
            metaclass = cfg.TRAIN.ALLCLASSES_FOURTH
        if args.meta_type == 5:
            metaclass = cfg.TRAIN.ALLCLASSES_FSODM

    # prepare meta sets for meta training
    if args.meta_train:
        # construct the input support datasets
        img_size = 224
        if args.phase == 1:
            img_set = "trainval"
        else:
            img_set = "train"

        if cfg.mask_on:
            metadataset = ImagePatchDiorMetaDataset("/home/hy/dataset/DIOR/",
                                                    img_set, metaclass, img_size, shots=shots, shuffle=True,
                                                    phase=args.phase)
        else:
            metadataset = DiorMetaDataset("/home/hy/dataset/DIOR/",
                                          img_set, metaclass, img_size, shots=shots, shuffle=True, phase=args.phase)

        metaloader = torch.utils.data.DataLoader(metadataset, batch_size=1, shuffle=False, num_workers=0,
                                                 pin_memory=True)

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name) 
    # filter roidb for the second phase
    if args.phase == 2:
        roidb = filter_class_roidb(roidb, args.shots, imdb)
        ratio_list, ratio_index = rank_roidb_ratio(roidb)
        imdb.set_roidb(roidb)

    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))
    sys.stdout.flush()

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers, pin_memory=False)

    # initilize the network here
    if args.net == 'Prototypecnn':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            meta_train=args.meta_train, meta_loss=args.meta_loss)
    fasterRCNN.create_architecture()

    # initilize the optimizer here
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        load_name = os.path.join(output_dir,
                                 '{}_metarcnn_{}_{}_{}.pth'.format(args.dataset, 200, #args.checksession, #得改成200才行
                                                                   args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        # the number of classes in second phase is different from first phase 
        if args.phase == 2:
            new_state_dict = OrderedDict()
            # initilize params of RCNN_cls_score and RCNN_bbox_pred for second phase
            RCNN_cls_score = nn.Linear(2048, imdb.num_classes) 
            RCNN_bbox_pred = nn.Linear(2048, 4 * imdb.num_classes)

            # Meta_cls_score = nn.Linear(2048, imdb.num_classes)

            for k, v in checkpoint['model'].items():
                name = k
                new_state_dict[name] = v
                if 'RCNN_cls_score.weight' in k:
                    new_state_dict[name] = RCNN_cls_score.weight
                if 'RCNN_cls_score.bias' in k:
                    new_state_dict[name] = RCNN_cls_score.bias
                if 'RCNN_bbox_pred.weight' in k:
                    new_state_dict[name] = RCNN_bbox_pred.weight
                if 'RCNN_bbox_pred.bias' in k:
                    new_state_dict[name] = RCNN_bbox_pred.bias
                # # 为了消融实验而增加的，到时候可以删除掉
                # if 'Meta_cls_score.weight' in k:
                #     new_state_dict[name] = Meta_cls_score.weight
                # if 'Meta_cls_score.bias' in k:
                #     new_state_dict[name] = Meta_cls_score.bias
            fasterRCNN.load_state_dict(new_state_dict)
        elif args.phase == 1:
            fasterRCNN.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']

        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.start_epoch, args.max_epochs):
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        meta_iter = iter(metaloader)
        for step in range(iters_per_epoch):
            try:
                data = next(data_iter)
            except:
                data_iter = iter(dataloader)
                data = next(data_iter)

            im_data_list = []
            im_info_list = []
            gt_boxes_list = []
            num_boxes_list = []

            # initilize the tensor holder here.
            im_data = torch.FloatTensor(1)
            im_info = torch.FloatTensor(1)
            num_boxes = torch.LongTensor(1)
            gt_boxes = torch.FloatTensor(1)
            # ship to cuda
            if args.cuda:
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                num_boxes = num_boxes.cuda()
                gt_boxes = gt_boxes.cuda()
            # make variable
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

            if args.meta_train:
                # get prn network input data
                try:
                    prndata,prncls = next(meta_iter)
                except:
                    meta_iter = iter(metaloader)
                    prndata, prncls = next(meta_iter)

                im_data_list.append(Variable(torch.cat(prndata,dim=0).cuda()))
                im_info_list.append(prncls)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])
                im_data_list.append(im_data)
                im_info_list.append(im_info)
                gt_boxes_list.append(gt_boxes)
                num_boxes_list.append(num_boxes)

            else:

                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])
                im_data_list.append(im_data)
                im_info_list.append(im_info)
                gt_boxes_list.append(gt_boxes)
                num_boxes_list.append(num_boxes)

            fasterRCNN.zero_grad()

            rois, rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, cls_prob, bbox_pred, meta_loss = fasterRCNN(im_data_list, im_info_list, gt_boxes_list,
                                                                    num_boxes_list)

            if args.meta_train:
                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + sum(RCNN_loss_cls) / args.batch_size + sum(
                    RCNN_loss_bbox) / args.batch_size + meta_loss / len(metaclass)
            else:
                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                       + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()
            loss.backward()
            # if args.net == "vgg16" or "res101":
            #     clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            torch.cuda.empty_cache()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval  # loss_temp is aver loss

                loss_rpn_cls = rpn_loss_cls.data[0]
                loss_rpn_box = rpn_loss_box.data[0]
                if not args.meta_train:
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]
                else:
                    loss_rcnn_cls = sum(RCNN_loss_cls) / args.batch_size
                    loss_rcnn_box = sum(RCNN_loss_bbox) / args.batch_size
                    loss_metarcnn = meta_loss / len(metaclass)

                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                if args.meta_train:
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, meta_loss %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_metarcnn ))
                else:
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                sys.stdout.flush()

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    niter = (epoch - 1) * iters_per_epoch + step
                    for tag, value in info.items():
                        writer.add_scalar(tag, value, niter)

                loss_temp = 0
                start = time.time()

        if args.meta_train:
            save_name = os.path.join(output_dir,
                                     '{}_{}_{}_{}_{}.pth'.format(str(args.dataset), str(args.net), shots, epoch,
                                                                 step))
        else:
            save_name = os.path.join(output_dir, '{}_{}_{}_{}.pth'.format(str(args.dataset), str(args.net),
                                                                          epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
    print('save model: {}'.format(save_name))
    end = time.time()
    print(end - start)

    if args.meta_train: # to extract the mean classes attentions of shots for testing
        class_attentions = collections.defaultdict(list)
        meta_iter = iter(metaloader)
        for i in range(shots):
            prndata, prncls = next(meta_iter)
            im_data_list = []
            im_info_list = []
            gt_boxes_list = []
            num_boxes_list = []
            im_data = torch.FloatTensor(1)
            if args.cuda:
                im_data = im_data.cuda()
            im_data = Variable(im_data, volatile=True)
            im_data.data.resize_(prndata.squeeze(0).size()).copy_(prndata.squeeze(0))
            im_data_list.append(im_data)
            attentions = fasterRCNN(im_data_list, im_info_list, gt_boxes_list, num_boxes_list,
                                            average_shot=True)
            for idx, cls in enumerate(prncls):
                class_attentions[int(cls)].append(attentions[idx])
        # calculate mean attention vectors of every class
        mean_class_attentions = {k: sum(v) / len(v) for k, v in class_attentions.items()}
        save_path = args.save_dir #'attentions'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(os.path.join(save_path, args.dataset + '_' +'split_{}'.format(args.meta_type) + '_' + 'mask_' + str(cfg.mask_on) + '_'  + 'attenRPN_' + str(cfg.RPN_Attention) + '_' + str(args.phase) + '_shots_' + str(args.shots) + '_mean_class_attentions.pkl'), 'wb') as f:
            pickle.dump(mean_class_attentions, f, pickle.HIGHEST_PROTOCOL)
        print('save ' + str(args.shots) + ' mean classes attentions done!')


        with open(os.path.join(save_path, args.dataset + '_' +'split_{}'.format(args.meta_type) + '_' + 'mask_' + str(cfg.mask_on) + '_'  + 'attenRPN_' + str(cfg.RPN_Attention) + '_' + str(args.phase) + '_shots_' + str(args.shots) + '_class_attentions.pkl'), 'wb') as f:
            pickle.dump(class_attentions, f, pickle.HIGHEST_PROTOCOL)
