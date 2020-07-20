# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:40:08 2020

@author: Jasmine
"""

import os
import openslide
import xml.etree.ElementTree as ET
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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


class Attention_92(nn.Module):
    def __init__(self):
        super(Attention_92,self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        
        self.dropout = nn.Dropout(p=0.5)

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3,20,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(20,50,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(50,60,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(60 * 8* 8, self.L),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = x .squeeze(0)

        H = self.feature_extractor_part1(x)
        #print(H.shape)
        #H = H.view(-1, 60 * 8 * 8)
        H = torch.flatten(H, start_dim =1)
        #print(H.shape)
        H = self.dropout(H)
        #print(H.shape)
        H = self.feature_extractor_part2(H)

        A = self.attention(H)
        A = torch.transpose(A,1,0)
        #print(A)
        A = F.softmax(A,dim=1)
        #print(A)

        M = torch.mm(A,H)
        M = self.dropout(M)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob,0.5).float()
        #print(Y_prob.shape, Y_hat.shape)

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat,_ = self.forward(X)
        #print(Y, Y_hat, Y_hat.eq(Y))
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, Y_prob
  
    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood =-1. * (Y * torch.log(Y_prob)+(1. - Y) * torch.log(1. - Y_prob))

        return neg_log_likelihood, A
    
    
    
    
class TSDataset_apply_overlap(data_utils.Dataset):
    def __init__(self,slidedir, patch_shape, bag_length_2D,transform):
        self.slidedir = slidedir
        self.patch_shape = patch_shape
        self.bag_length_2D = bag_length_2D
        self.transform = transform

        self.bag_shape = (int(self.patch_shape[0]*self.bag_length_2D[0]), int(self.patch_shape[1]*self.bag_length_2D[1]))
        self.slide = openslide.OpenSlide(slidedir)
        self.slide_shape = self.slide.dimensions
        self.grid, self.grid_num = self.create_grid()
        print(self.slide_shape)
        #print(self.grid)
        
    
    def create_grid(self):
        grid = []
        num_grid_x = self.slide_shape[0] // self.patch_shape[0] - self.bag_length_2D[0]
        num_grid_y = self.slide_shape[1] // self.patch_shape[1] - self.bag_length_2D[1]
        print(num_grid_x,num_grid_y)
        for i in range(num_grid_x):
            for j in range(num_grid_y):
                upperLeft_bag_level = (int(i*self.patch_shape[0]), int(j*self.patch_shape[1]))
                upperLeft_instance_level = []
                for instance_i in range(self.bag_length_2D[0]):
                    for instance_j in range(self.bag_length_2D[1]):
                        upperLeft_instance_level.append((int(instance_i*self.patch_shape[0] + upperLeft_bag_level[0]), 
                                                         int(instance_j*self.patch_shape[1] + upperLeft_bag_level[1])))
                upperLeft={
                    'bag':upperLeft_bag_level,
                    'instance':upperLeft_instance_level
                }
                grid.append(upperLeft)
        return grid,(num_grid_x,num_grid_y)
    
    def pack_bag(self, upperLeft):
        instance_level = upperLeft['instance']
        Patches = []
        for upperLeft_patch in instance_level:
            #print(upperLeft_patch)
            patch = self.slide.read_region(upperLeft_patch,0,self.patch_shape)
            patch = np.array(patch)[:,:,:3]
            if self.transform is not None:
                patch = self.transform(patch)
            Patches.append(patch)
        bag = np.stack(Patches, axis=0)
        return bag
    
    def __len__(self):
        return len(self.grid)
    
    def __getitem__(self,idx):
        upperLeft = self.grid[idx]
        bag = self.pack_bag(upperLeft)
        return bag, upperLeft
    
    

class Args:
    def __init__(self):
        self.root_dir = "~/ashel-slide/458599"
        self.output_dir='~/ashel-slide/458599/a603'
        self.slidedir='~/ashel-slide/458603.svs'
        self.annotdir='~/ashel-slide/458603.xml'
        self.patch_shape = (92,92)
        self.model = Attention_92()
        self.model_path = "model_92_100.pth"
        self.bag_length_2D = (10,10)
        self.overlap = True
        
args = Args()

device = torch.device("cuda")
model = args.model
model.load_state_dict(torch.load(os.path.join(args.root_dir, args.model_path)))
model.to(device)


if args.overlap:
    apply_set = TSDataset_apply_overlap( slidedir = args.slidedir,
                               patch_shape = args.patch_shape, 
                               bag_length_2D = args.bag_length_2D,
                               transform = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.777, 0.778, 0.769),
                                                                                  (0.124,0.128,0.128))])
                              )

    apply_loader = data_utils.DataLoader(apply_set, batch_size = 1, shuffle = False)
    print(len(apply_loader))
    for bag_idx, (bag, upperLeft) in enumerate(apply_loader):
        if bag_idx > 2:
            break
        print(bag.shape)
    #print(upperLeft)
    
    hot_map_instance = np.zeros((args.bag_length_2D[1] + apply_set.grid_num[1], args.bag_length_2D[0] + apply_set.grid_num[0]))
    hot_map_bag = np.zeros((args.bag_length_2D[1] + apply_set.grid_num[1], args.bag_length_2D[0] + apply_set.grid_num[0]))
    hot_map_count = np.zeros((args.bag_length_2D[1] + apply_set.grid_num[1], args.bag_length_2D[0] + apply_set.grid_num[0]))
    print("Shape of hot map is ", hot_map_instance.shape)

    for batch_idx, (data, upperleft) in enumerate(apply_loader):
        data = data.cuda()
        data = Variable(data)
        Y_prob, predicted_label, attention_weights = model.forward(data)
        Y_prob = Y_prob.cpu().data.numpy()[0][0]
        predicted_label = predicted_label.cpu().numpy()[0][0]
        attention_weights = attention_weights.cpu().data.numpy()[0].tolist()
    
        print(batch_idx, Y_prob,predicted_label)
    
        instance_level = upperleft['instance'] 
        bag_level = upperleft['bag']
        instance_level = [(i[0]//args.patch_shape[0],i[1]//args.patch_shape[1]) for i in instance_level]
        (x_bag, y_bag) = (bag_level[0]//args.patch_shape[0], bag_level[1]//args.patch_shape[1])
    
        hot_map_bag[y_bag:y_bag + args.bag_length_2D[1], x_bag: x_bag+args.bag_length_2D[0]] += Y_prob
        hot_map_count[y_bag:y_bag + args.bag_length_2D[1], x_bag: x_bag+args.bag_length_2D[0]] += 1
    
        for idx, (x, y) in enumerate(instance_level):
            hot_map_instance[y, x] += attention_weights[idx]

    #print(np.max(hot_map_instance), np.min(hot_map_instance))
    #print(np.max(hot_map_bag), np.min(hot_map_bag))

hot_map_instance /= hot_map_count
hot_map_bag /= hot_map_count

np.save(os.path.join(args.output_dir,'hot_map_instance.npy'), hot_map_instance)
np.save(os.path.join(args.output_dir,'hot_map_bag.npy'), hot_map_bag)


"""
Display heatmap
"""

import matplotlib.cm as cm
import matplotlib.pylab as pylab
params = {'legend.fontsize': 28,
          'figure.figsize': (40, 20),
         'axes.labelsize': 'x-large',
         'axes.titlesize':36,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

temp = np.mean(hot_map_instance[hot_map_bag>0.5])
vmin = temp*0.5
vmax = temp*2.0
hot_map_instance_01 = (hot_map_instance-vmin)/(vmax-vmin)
cmap = cm.get_cmap()
RGBA = cmap(hot_map_instance_01)
RGBA[...,-1]= hot_map_bag

f,ax = plt.subplots(figsize=(40,20))
h = ax.imshow(RGBA)
cb = plt.colorbar(h)
ticks = cb.get_ticks()
cb.set_ticks(ticks)
cb.set_ticklabels(ticks*(vmax-vmin) + vmin)
ax.set_title("Hot map")
f.show()
plt.savefig(os.path.join(args.output_dir,"hot_map.png"))


def read_annotation(annotdir):
    tree = ET.parse(annotdir)
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
    return points_arr_tumor, points_arr_stroma


annotdir = args.annotdir
points_arr_tumor, points_arr_stroma = read_annotation(annotdir)
tumor_x = [i[0]//92 for i in points_arr_tumor ]
tumor_y = [i[1]//92 for i in points_arr_tumor ]
stroma_x = [i[0]//92 for i in points_arr_stroma ]
stroma_y = [i[1]//92 for i in points_arr_stroma ]

slidedir = args.slidedir
slide_ob = openslide.OpenSlide(slidedir)
wsi = slide_ob.get_thumbnail((slide_ob.dimensions[0]//92,slide_ob.dimensions[1]//92))
wsi = np.array(wsi)
f,ax = plt.subplots(figsize=(40,20))
ax.imshow(wsi)
ax.scatter(tumor_x,tumor_y, c='r',label = 'tumor')
ax.scatter(stroma_x,stroma_y, c='g',label = 'stroma')
ax.legend()
h = ax.imshow(RGBA)
cb = plt.colorbar(h)
ticks = cb.get_ticks()
cb.set_ticks(ticks)
cb.set_ticklabels(ticks*(vmax-vmin) + vmin)
ax.set_title("WSI + Hot map + Annotation")
f.show()
plt.savefig(os.path.join(args.output_dir,"hot_map_wsi.png"))

wsi_hr = slide_ob.get_thumbnail((slide_ob.dimensions[0]//10,slide_ob.dimensions[1]//10))
wsi_hr = np.array(wsi_hr)
tumor_x_hr = [i[0]//10 for i in points_arr_tumor ]
tumor_y_hr = [i[1]//10 for i in points_arr_tumor ]
stroma_x_hr = [i[0]//10 for i in points_arr_stroma ]
stroma_y_hr = [i[1]//10 for i in points_arr_stroma ]
f,ax = plt.subplots(figsize=(40,20))
ax.imshow(wsi_hr)
ax.scatter(tumor_x_hr,tumor_y_hr, c='r',label = 'tumor')
ax.scatter(stroma_x_hr,stroma_y_hr, c='g',label = 'stroma')
ax.legend()
ax.set_title("WSI + Annotation")
f.show()
plt.savefig(os.path.join(args.output_dir,"wsi.png"))