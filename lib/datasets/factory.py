# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}

from lib.datasets.coco import coco
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.dior import dior
# # Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test','shots', 'train_first_split', 'train_second_split', 'train_third_split']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))


for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

for year in ['2017']:
  for split in ['train', 'val']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# set dior datasets
for split in ['train', 'val', 'trainval', 'test', 'shots', 'train_first_split', 'train_second_split', 'train_third_split', 'train_easy_novel_split', 
              'train_split_1', 'train_split_2', 'train_split_3', 'train_split_4', 'train_split_11', 'train_split_22', 'train_split_33', 'train_split_44',
              'train_ablation_split_11', 'train_ablation_split_22', 'train_ablation_split_33', 'train_ablation_split_44', 'train_split_55']:
  name = 'dior_{}'.format(split)
  __sets[name] = (lambda split=split: dior(split))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
