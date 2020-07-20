# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 17:06:54 2020

@author: Jasmine
"""

import os
os.environ['PATH']="D:/openslide-win64-20171122/bin"+';'+os.environ['PATH']
import openslide
import xml
import xml.etree.ElementTree as ET
import PIL
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from skimage import measure
from skimage.transform import resize
import numpy as np


import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import imageio


class TSDataset_save(data_utils.Dataset):
    def __init__(self,slidedir, annotdir, patch_shape, energy_threshold):
        self.slidedir = slidedir
        self.annotdir = annotdir
        self.patch_shape = patch_shape
        self.energy_threshold = energy_threshold
        
        self.slide = openslide.OpenSlide(slidedir)
        self.slide_shape = self.slide.dimensions
        self.annot_mask = self.read_annotation()
        self.grid = self.create_grid()
        #print(self.grid)
        
        
    def read_annotation(self):
        tree = ET.parse(self.annotdir)
        root = tree.getroot()
        points_arr_tumor = []
        points_arr_stroma = []
        for r in root.iter('Annotation'):
            if r.attrib['Id']== '1':
                for v in r.iter('Vertex'):
                    points_arr_tumor.append((int(v.attrib['X']), int(v.attrib['Y'])))
            if r.attrib['Id']== '2':
                for v in r.iter('Vertex'):
                    points_arr_stroma.append((int(v.attrib['X']), int(v.attrib['Y'])))
        points_tumor = tuple(points_arr_tumor)
        points_stroma = tuple(points_arr_stroma)
        mask =  Image.new('1', self.slide_shape)
        draw = ImageDraw.Draw(mask)
        draw.polygon(points_tumor, fill=1, outline=1)
        draw.polygon(points_stroma, fill=2, outline=2)
      
        return mask
    
    def create_grid(self):
        grid = []
        num_grid_x = self.slide_shape[0] // self.patch_shape[0]
        num_grid_y = self.slide_shape[1] // self.patch_shape[1]
        for i in range(num_grid_x):
            for j in range(num_grid_y):
                upperLeft = (int(i*self.patch_shape[0]), int(j*self.patch_shape[1]))
                grid.append(upperLeft)
        #print(grid)
        #print(len(grid))
        return grid
    
    def energy(self, patch):
        energy_patch = 0
        for i in range(3):
            energy_patch += np.mean(np.square(np.gradient(patch[:,:,i])))
        energy_patch /= 3
        #print(energy_patch)
        return energy_patch
    
    def __len__(self):
        print(type(self.grid))
        return len(self.grid)
    
    def __getitem__(self,idx):
        upperLeft = self.grid[idx]
        patch = self.slide.read_region(upperLeft,0,self.patch_shape)
        patch = np.array(patch)[:,:,:3]
        mask_patch = self.annot_mask.crop((upperLeft[0],upperLeft[1],
                                           upperLeft[0]+self.patch_shape[0],
                                           upperLeft[1]+self.patch_shape[1]))
        if mask_patch.histogram()[1] >= 0.9 * self.patch_shape[0]*self.patch_shape[1]:
            label_region = 1
        elif mask_patch.histogram()[2] >= 0.9 * self.patch_shape[0]*self.patch_shape[1]:
            label_region = 2
        else:
            label_region = 0
        
        if  self.energy(patch)  <= self.energy_threshold:
            label_energy = 0
        else:
            label_energy = 1
        return patch, [label_region, label_energy]
    
class Args():
    def __init__(self):
        self.slidedir = "D:/Ashel-slide/458608.svs"
        self.annotdir = "D:/Ashel-slide/458608.xml"
        self.save_root_dir = "D:/Ashel-slide/458608"
        self.patch_shape = (92,92)
args = Args()   


patch_set = TSDataset_save(slidedir = args.slidedir, 
                      annotdir = args.annotdir, 
                      patch_shape = args.patch_shape, 
                      energy_threshold = 7000)
patch_loader = data_utils.DataLoader(patch_set, batch_size = 1, shuffle = False)
print("There are {} patches in all".format(len(patch_set))) 


save_root_dir = os.path.join(args.save_root_dir,str(args.patch_shape[0]))
save_tumor_dir = os.path.join(save_root_dir,'tumor')
save_stroma_dir = os.path.join(save_root_dir,'stroma')
#save_other_dir = os.path.join(save_root_dir,'other')
save_tumor_blank = os.path.join(save_tumor_dir,'blank')
save_tumor_non_blank = os.path.join(save_tumor_dir,'non-blank')
save_stroma_blank = os.path.join(save_stroma_dir,'blank')
save_stroma_non_blank = os.path.join(save_stroma_dir,'non-blank')
os.makedirs(save_tumor_blank)
os.makedirs(save_tumor_non_blank)
os.makedirs(save_stroma_blank)
os.makedirs(save_stroma_non_blank)
#os.makedirs(save_other_dir )


number_useful = 0
for batch_idx, (data, label) in enumerate(patch_loader):
    label_region = label[0].numpy()[0]
    label_energy = label[1].numpy()[0]
    patch = data.numpy()[0,:,:,:]
    print(batch_idx,label_region, label_energy)
    # Choose the right subdirectory to write into
    if label_region == 0 :
        continue
        #savedir = save_other_dir
    elif label_region == 1:
        if label_energy == 0:
            savedir = save_tumor_blank
        else:
            savedir = save_tumor_non_blank
            number_useful +=1
    else:
        if label_energy == 0:
            savedir = save_stroma_blank
        else:
            savedir = save_stroma_non_blank
            number_useful +=1
    patch_to_save = Image.fromarray(patch)
    patch_to_save.save(os.path.join(savedir,str(batch_idx)+'.png'))   
    
print("There are {} useful patches in all".format(number_useful))