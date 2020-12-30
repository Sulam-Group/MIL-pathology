# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 17:06:54 2020

@author: Jasmine
"""

import os
import openslide
import xml
import xml.etree.ElementTree as ET
import PIL
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import seaborn as sns
import imageio
import argparse
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import pandas as pd
import cv2 as cv

    
ap = argparse.ArgumentParser()
ap.add_argument("data_dir",help = "The path to the data dir")
ap.add_argument("slide_ID", help = "Which slide you would like to crop patches")
ap.add_argument("patch_shape", help = "The shape of patch")
ap.add_argument("stride", help = "Stride when cropping patches, unit is one patch length")
ap.add_argument('layer', help = "Wether or not to downsample the slide")
args_io = ap.parse_args()

class Args():
    def __init__(self):
        self.slidedir = args_io.data_dir + "/" + args_io.slide_ID + ".svs"
        self.annotdir = args_io.data_dir + "/" +  args_io.slide_ID + ".xml"
        self.save_root_dir = "results/grid_"+ args_io.patch_shape  + '/' + args_io.slide_ID 
        self.patch_shape = (int(args_io.patch_shape),int(args_io.patch_shape))
        self.stride = float(args_io.stride)
        self.layer = int(args_io.layer)
args = Args()   

if not os.path.exists(args.save_root_dir):
    os.makedirs(args.save_root_dir)
    
class TSpatches_save(data_utils.Dataset):
    def __init__(self,slidedir, annotdir, patch_shape, stride, layer, transform = None):
        self.slidedir = slidedir
        self.annotdir = annotdir
        self.patch_shape = patch_shape
        self.stride = stride
        self.layer = layer
        self.transform = transform
        
        self.slide = openslide.OpenSlide(slidedir)
        self.slide_shape = self.slide.level_dimensions[self.layer]
        if os.path.exists(self.annotdir):
            self.annot_mask = self.read_annotation()
        else:
            self.annot_mask = None
        self.tissue_mask = self.create_tissue_mask()
        self.grid, self.num_grid_x,self.num_grid_y= self.create_grid()
        print(self.slide.level_downsamples[self.layer])
        
    def read_annotation(self):
        tree = ET.parse(self.annotdir)
        root = tree.getroot()
        points_arr_tumor = []
        points_arr_stroma = []
        for r in root.iter('Annotation'):
            if r.attrib['Id']== '1':
                for v in r.iter('Vertex'):
                    points_arr_tumor.append((int(v.attrib['X'])/self.slide.level_downsamples[self.layer], 
                                             int(v.attrib['Y'])/self.slide.level_downsamples[self.layer]))
            if r.attrib['Id']== '2':
                for v in r.iter('Vertex'):
                    points_arr_stroma.append((int(v.attrib['X'])/self.slide.level_downsamples[self.layer], 
                                              int(v.attrib['Y'])/self.slide.level_downsamples[self.layer]))

        mask =  Image.new('1', self.slide_shape)
        draw = ImageDraw.Draw(mask)
        if len(points_arr_tumor)>0:
            points_tumor = tuple(points_arr_tumor)
            draw.polygon(points_tumor, fill=1, outline=1)
        if len(points_arr_stroma)>0:
            points_stroma = tuple(points_arr_stroma)
            draw.polygon(points_stroma, fill=2, outline=2)      
        return mask
       
    def create_grid(self):
        grid = []
        num_grid_x = self.slide_shape[0] // (int(self.patch_shape[0] * self.stride))
        num_grid_y = self.slide_shape[1] // (int(self.patch_shape[1] * self.stride))
        print(num_grid_x, num_grid_y)
        for i in range(num_grid_x):
            for j in range(num_grid_y):
                upperLeft = (int(i*self.patch_shape[0] * self.stride), int(j*self.patch_shape[1] * self.stride))
                grid.append(upperLeft)
        return grid,num_grid_x,num_grid_y

    def create_tissue_mask(self):
        width,height = self.slide.dimensions
        thumbnail = np.array(self.slide.get_thumbnail((width//self.patch_shape[0],height//self.patch_shape[1])))
        thumbnail = cv.cvtColor(thumbnail,cv.COLOR_RGB2GRAY)
        _,mask = cv.threshold(thumbnail,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask = abs(mask-255)
        print(np.sum(mask),mask.shape)
        mask = cv.resize(mask,(self.slide_shape[0],self.slide_shape[1]))
        print(np.sum(mask),mask.shape)
        return mask
    
    def __len__(self):
        return len(self.grid)
    
    def __getitem__(self,idx):
        upperLeft = self.grid[idx]
        upperLeft_layer0 = (int(upperLeft[0]*self.slide.level_downsamples[self.layer]),int(upperLeft[1]*self.slide.level_downsamples[self.layer]))
        patch = self.slide.read_region(upperLeft_layer0,self.layer,self.patch_shape)
        patch = np.array(patch)[:,:,:3]
        if self.transform is not None:
            patch = self.transform(patch)
        if self.annot_mask:
            AnnotMask_patch = self.annot_mask.crop((upperLeft[0],upperLeft[1],
                                           upperLeft[0]+self.patch_shape[0],
                                           upperLeft[1]+self.patch_shape[1]))

            if AnnotMask_patch.histogram()[1] == 1 * self.patch_shape[0]*self.patch_shape[1]:
                label_region = 1   # tumor
            elif AnnotMask_patch.histogram()[2] == 1 * self.patch_shape[0]*self.patch_shape[1]:
                label_region = 2  # stroma
            elif AnnotMask_patch.histogram()[0] == 1* self.patch_shape[0]*self.patch_shape[1]:
                label_region = 0 # outside annotation
            else:
                label_region = 3   # intersection
        else:
            label_region = np.nan
        
        TissueMask_patch = self.tissue_mask[upperLeft[1]:upperLeft[1]+self.patch_shape[1], upperLeft[0]:upperLeft[0]+self.patch_shape[0]]
        #print(np.sum(TissueMask_patch))
        if np.sum(TissueMask_patch) >= 0.5 * self.patch_shape[0]*self.patch_shape[1]:
            label_tissue = 1  # tissue
        else:
            label_tissue = 0 # blank

        return patch, upperLeft, label_region,label_tissue
    
    
    
patch_set = TSpatches_save(slidedir = args.slidedir, 
                      annotdir = args.annotdir, 
                      patch_shape = args.patch_shape,
                      stride = args.stride,
                      layer = args.layer)
patch_loader = data_utils.DataLoader(patch_set, batch_size = 1, shuffle = False)
print("There are {} patches in all".format(len(patch_set))) 

Tile_array = np.zeros((patch_set.num_grid_y, patch_set.num_grid_x,2)) #[:,:,0]:annotation mask; [:,:,1]: tissue mask
for batch_idx, (patch, upperLeft, label_region,label_tissue) in enumerate(patch_loader):
    label_region = label_region.numpy()[0]
    label_tissue = label_tissue.numpy()[0]
    upperLeft_x = upperLeft[0].numpy()[0]
    upperLeft_y = upperLeft[1].numpy()[0]
    patch = patch.numpy()[0,:,:,:]   
    col_id = batch_idx // patch_set.num_grid_y
    row_id = batch_idx % patch_set.num_grid_y
    Tile_array[row_id,col_id,0]= label_region
    Tile_array[row_id,col_id,1]= label_tissue 
    if label_tissue==1:
        print(batch_idx, label_region, label_tissue, upperLeft_x, upperLeft_y)
        patch_to_save = Image.fromarray(patch)
        patch_to_save.save(os.path.join(args.save_root_dir,str(row_id)+'_'+str(col_id)+'.png'))

with open('results/tile_'+args_io.slide_ID+'_'+args_io.patch_shape+'_'+args_io.stride+'_'+args_io.layer+'.npy','wb') as f:
    np.save(f,Tile_array)
f.close()

