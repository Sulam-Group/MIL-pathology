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
import argparse
import random
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from datetime import datetime
import argparse
import pickle
import cv2 as cv
import pandas as pd
######################################################################### functions #############################################################################
# Help function so that we can pass "true" or "false" 
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Identify tissues
def HSV_otsu(img,channels=['H','S']):
    img = np.array(img)[:,:,:3]
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    if 'H' in channels:
        _,mask_H = cv.threshold(hsv[:,:,0],0,179,cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask_H = mask_H != 0
    else:
        mask_H = np.ones((img.shape[0],img.shape[1]),dtype=bool)
    if 'S' in channels:
        _,mask_S = cv.threshold(hsv[:,:,1],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask_S = mask_S != 0
    else:
        mask_S = np.ones((img.shape[0],img.shape[1]),dtype=bool)
    if 'V' in channels:
        _,mask_V = cv.threshold(hsv[:,:,2],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        mask_V = mask_V != 0
    else:
        mask_V = np.ones((img.shape[0],img.shape[1]),dtype=bool)
    mask = np.logical_and(mask_H,mask_S,mask_V).astype(np.uint8)
    return mask
def GRAY_otsu(img,histeq=False):
    img = np.array(img)[:,:,:3]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    if histeq:
        clahe =  cv.createCLAHE(clipLimit=40, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    otsu_threshold,mask = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    mask = (mask == 0).astype(np.uint8)
    return mask
def binary_PAIP(img,threshold = [235, 210, 235]):
    mask_R = img[:,:,0]<threshold[0]
    mask_G = img[:,:,1]<threshold[1]
    mask_B= img[:,:,2]<threshold[2]
    mask = np.logical_and(mask_R,mask_G,mask_B).astype(np.uint8)
    return mask

# Crop boundary
def crop_boundary(img, boundary_ratio=[0,0,0,0]):
    # boundary_ratio: 
    boundary_ratio_u = boundary_ratio[0]
    boundary_ratio_b = boundary_ratio[1]
    boundary_ratio_l = boundary_ratio[2]
    boundary_ratio_r = boundary_ratio[3]
    NROW,NCOL = img.shape
    img[:int(NROW*boundary_ratio_u),:]=0
    if boundary_ratio_b!=0:
        img[-int(NCOL*boundary_ratio_b):,:]=0
    img[:,:int(NROW*boundary_ratio_l)]=0
    if boundary_ratio_r!=0:
        img[:,-int(NCOL*boundary_ratio_r):]=0
    return img

# MIL model
class Attention_modern(nn.Module):
    def __init__(self,cnn,focal_loss=False):
        super(Attention_modern,self).__init__()
        self.L = 1000
        self.D = 64
        self.K = 1 
        self.focal_loss = focal_loss     
        self.feature_extractor = cnn      
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K))
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = x .squeeze(0)
        H = self.feature_extractor(x)
#         H = H.logits
        A = self.attention(H)
        A = torch.transpose(A,1,0)
        A = F.softmax(A,dim=1)

        M = torch.mm(A,H)
        Y_prob = self.classifier(M)[0]
        Y_hat = torch.ge(Y_prob,0.5).float()
        return Y_prob, Y_hat, A
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat,_ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, Y_prob  
    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        if not self.focal_loss:
            neg_log_likelihood =-1. * (Y * torch.log(Y_prob)+(1. - Y) * torch.log(1. - Y_prob))
        else:
            #print(Y.cpu().data.numpy())
            if Y.cpu().data.numpy()[0]==0:
                #print(Y.cpu().data.numpy()[0])
                Y_prob = 1-Y_prob            
            if Y_prob.cpu().data.numpy()[0]<0.2:
                gamma = 5
            else:
                gamma = 3
            neg_log_likelihood =-1. *(1-Y_prob)**gamma* torch.log(Y_prob)
        return neg_log_likelihood, A

# Load pre-trained feature encoder
def load_vgg16():
    vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    num_layer = 0
     # freeze parameters in VGG encoder, since the training data is compartaively small
    for child in vgg.children():
        num_layer+=1
        if num_layer < 3:
            for param in child.parameters():
                param.requires_grad = False  
    return vgg

# Singleton MIL bag construction
class TSbags_apply_neighbor(data_utils.Dataset):
    def __init__(self,slide_dir,ROW,COL,patch_shape, unit, transform = None):
        self.slide = openslide.OpenSlide(slide_dir)
        self.ROW, self.COL = ROW, COL
        self.patch_shape = patch_shape
        self.unit = unit
        self.transform = transform 
        self.block_shape_width = 1
        self.block_shape_height = 1
        self.stride = 1
        self.block_to_unit = self.patch_shape//self.unit
        self.patch_to_unit = self.patch_shape//self.unit 
    def pack_one_bag(self, row, col):
        upperLeft_bag_x = col * self.unit+self.unit/2-(self.block_to_unit*self.unit)/2
        upperLeft_bag_y = row * self.unit+self.unit/2-(self.block_to_unit*self.unit)/2
        Patches_in_one_bag = []
        Patches_index_in_one_bag = []
        for i in range(self.block_shape_height):
            for j in range(self.block_shape_width):
                upperLeft_patch_x = int(upperLeft_bag_x + j*self.patch_shape)
                upperLeft_patch_y = int(upperLeft_bag_y + i*self.patch_shape)
                patch = Image.fromarray(np.array(self.slide.read_region((upperLeft_patch_x,upperLeft_patch_y),0,(self.patch_shape,self.patch_shape)))[:,:,:3])
                if self.transform is not None:
                    patch = self.transform(patch)
                Patches_in_one_bag.append(patch)
                Patches_index_in_one_bag.append((i,j))
        bag = np.stack(Patches_in_one_bag, axis=0)
        return bag,Patches_index_in_one_bag
    
    def __len__(self):
        return len(self.ROW)
    
    def __getitem__(self,idx):
        row_center_bag, col_center_bag = self.ROW[idx], self.COL[idx]
        bag,Patches_index_in_one_bag = self.pack_one_bag(row_center_bag, col_center_bag)
        return bag, row_center_bag, col_center_bag, Patches_index_in_one_bag

# Training
def train(loader, model, optimizer, scheduler):
    model.train()
    train_loss = 0.
    train_error = 0.
    optimizer.zero_grad()
    for batch_idx, (data, label) in enumerate(loader):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, _ = model.calculate_objective(data, bag_label)
        error, _, _ =model.calculate_classification_error(data, bag_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.data[0]
        train_error += error    
        del data
        del bag_label
    train_loss /= len(loader)
    train_error /= len(loader)
    return 1 - train_error, train_loss.cpu().numpy()
######################################################################### functions ############################################################################# 
     
ap = argparse.ArgumentParser()
ap.add_argument('slide_root',help = "The location of the whole-slide image.")
ap.add_argument('slide_ID',help = "Name of the whole-slide image.")
ap.add_argument('slide_format',help = "Dataformat the whole-slide image. Permitted format can be `.svs`, `,ndpi`, and `.tif`.")
ap.add_argument('model_dir',help = "The path to the MIL model")
ap.add_argument('heatmap_save_root',help = "Where the predicted heatmap will be saved")
ap.add_argument("--remove_blank", type = int, help = "How to remove blank regions (i.e. identify tissue regions) of WSI. We provide three functions: 0: convert to HSV and then OTSU threshold on H and S channel; 1: apply [235, 210, 235] on RGB channel, respecitively; 2: convert to gray image, then OTSU threshold.  Default is 0. For new dataset, the user is encouraged to write customed function",default = 0)
ap.add_argument("--length_bag_mean",type = int, help = "Average length of bag (Binomial distribution),default = 10",default = 10)
ap.add_argument("--num_bags",type = int, help = "Number of bags to train,default = 1000", default = 1000)
ap.add_argument("--focal_loss", type = str2bool, help = "Whether or not to use focal loss (True: using focal loss; Flase: using cross entropy), default is FL", default=True)
ap.add_argument("--patch_shape",type = int, help = "Patch shape(size), default is 256", default=256)
ap.add_argument("--unit",type = int, help = "Samllest unit when cropping patches, default is 256", default=256)
ap.add_argument("--gpu", help = "gpu",default = '0')
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
loader_kwards = {'num_workers':1, 'pin_memory':True} 


print('-------------------------Read slide-------------------------------')
slide_ob = openslide.OpenSlide(args.slide_root+'/'+args.slide_ID+args.slide_format)
     
print('-------------------------Create masks (tissue mask )------------------------------')   
# Maks are all loww-resolution versions, downsampled by args.unit
boundary_ratio = [0,0,0,0] # [a,b,c,d] means cropping a * height, b * height, c * width, d * width from the top, bottom, left and right sides of the WSI. It can be useful if their are some smear at the boarder of slides
thumbnail = np.array(slide_ob.get_thumbnail((slide_ob.dimensions[0]/args.unit,slide_ob.dimensions[1]/args.unit)))[:,:,:3] # thumbnail 
# generate tissue mask to exclude blank regions in the WSI
if args.remove_blank == 0:
    mask_tissue = crop_boundary(HSV_otsu(thumbnail),boundary_ratio)
elif args.remove_blank == 1:
    mask_tissue = crop_boundary(binary_PAIP(thumbnail),boundary_ratio)
else:
    mask_tissue = crop_boundary(GRAY_otsu(thumbnail,True),boundary_ratio)
mask_tissue = cv.resize(mask_tissue,(int(slide_ob.dimensions[0]/args.unit),int(slide_ob.dimensions[1]/args.unit))) 
print("----------------------------------Load in model-----------------------------------------------")
device = torch.device("cuda")
feature_extractor = 'vgg'
model = Attention_modern(load_vgg16()) # Model artichecture
model.load_state_dict(torch.load(args.model_dir)) # Load in the pre-trained model
model.to(device)                                
print('----------------------------------Create "Singleton bags"----------------------------------------')    
# singleton bag: one patch is a bag
ROW, COL = np.where(mask_tissue==1)
patch_to_unit = args.patch_shape//args.unit
apply_set = TSbags_apply_neighbor(
                                  slide_dir = args.slide_root + '/'+args.slide_ID + args.slide_format,
                                  ROW=ROW,
                                  COL=COL,
                                  patch_shape = args.patch_shape,
                                  unit = args.unit,
                                  transform = transforms.Compose([
                                  transforms.Resize(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
apply_loader = data_utils.DataLoader(apply_set, batch_size = 1, shuffle = False)

print('----------------------------------Applying the MIL model to WSI----------------------------------------')    
# initialize a heatmap to save predicted scores for each patch
hot_map_bag = np.zeros_like(mask_tissue,dtype=float)
hot_map_bag[:] = np.nan
for batch_idx, (bag, row_center_bag, col_center_bag, Patches_index_in_one_bag) in enumerate(apply_loader):
    try:
        bag = bag.to(device=device, dtype=torch.float)
        bag = Variable(bag,requires_grad=False)
        Y_prob, predicted_label, attention_weights = model.forward(bag)
        Y_prob = Y_prob.cpu().data.numpy()[0]
        predicted_label = predicted_label.cpu().numpy()[0]
        attention_weights = attention_weights.cpu().data.numpy()[0] 
        del bag
        print(batch_idx,'/',len(apply_set),Y_prob,predicted_label) 
        row_center_bag = row_center_bag.cpu().data.numpy()[0]
        col_center_bag = col_center_bag.cpu().data.numpy()[0]
        hot_map_bag[row_center_bag,col_center_bag] = Y_prob
    except:
        continue

print('-------------------------Saving heatmap-------------------------------')
np.save(args.heatmap_save_root+'/heatmap_'+args.slide_ID+'.npy',hot_map_bag)

