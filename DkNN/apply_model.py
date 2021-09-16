from sklearn.cluster import KMeans
import os
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
import seaborn as sns
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import imageio
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import random
from datetime import datetime
import pickle
import torchvision.models as models
from torchsummary import summary
import cv2 as cv
from scipy.stats import binom
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.stats import binom
from sklearn.neighbors import NearestNeighbors
import faiss

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

# Generate a binary label mask based on the corrdinates of tumor boundaries
def create_tumor_mask(slide_ob,Annotations,downsample_scale):
    mask =  Image.new('1', (int(np.round(slide_ob.dimensions[0]/downsample_scale)),int(np.round(slide_ob.dimensions[1]/downsample_scale))))
    draw = ImageDraw.Draw(mask)
    for i in range(len(Annotations[0])):
        Annotation = Annotations[0][i]
        Annotation = [(i[0]/downsample_scale,i[1]/downsample_scale) for i in Annotation]
        draw.polygon(Annotation,fill=1,outline=0)
    for i in range(len(Annotations[1])):
        Annotation = Annotations[1][i]
        Annotation = [(i[0]/downsample_scale,i[1]/downsample_scale) for i in Annotation]
        draw.polygon(Annotation,fill=0,outline=0)
    #mask = np.array(mask.resize((int(slide_ob.dimensions[0]/downsample_scale), int(slide_ob.dimensions[1]/downsample_scale))))
    mask = np.array(mask)
    mask = (mask == 1).astype(np.uint8)
    return mask

# Dataset for individual patches (instead of the MIL bags)   
class TSpatch_apply(data_utils.Dataset):
    def __init__(self, slide_dir, unit, patch_shape, ROW, COL,transform):
        self.slide = openslide.OpenSlide(slide_dir)
        self.width, self.height = self.slide.dimensions
        self.unit = unit
        self.patch_shape = patch_shape
        self.ROW = ROW
        self.COL = COL
        self.transform = transform
          
    def __len__(self):
        return len(self.ROW)
    
    def __getitem__(self, index):
        row, col = self.ROW[index], self.COL[index]
        upperLeft_x, upperLeft_y = int(col * self.unit + self.unit/2), int(row * self.unit + self.unit/2)        
        patch = self.slide.read_region((upperLeft_x, upperLeft_y),0,(self.patch_shape,self.patch_shape))
        patch = Image.fromarray(np.array(patch)[:,:,:3])
        if self.transform is not None:
            patch = self.transform(patch)
        return patch,row,col 
    
# A binary classifier adjusted based on VGG
class vgg_binary(nn.Module):
    def __init__(self,vgg):
        super(vgg_binary,self).__init__()
        
        self.feature_extractor = vgg
        
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )      
    def forward(self,x):
        H = self.feature_extractor(x)
        Y_prob = self.classifier(H)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        Y_hat = torch.ge(Y_prob,0.5).float()
        return Y_prob, Y_hat
# auxillary function for the binary classifier
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x  
######################################################################### functions #############################################################################
ap = argparse.ArgumentParser()
ap.add_argument('slide_root',help = "The location of the whole-slide image.")
ap.add_argument('slide_ID',help = "Name of the whole-slide image.")
ap.add_argument('slide_format',help = "Dataformat the whole-slide image. Permitted format can be `.svs`, `,ndpi`, and `.tif`.")
ap.add_argument('ca_path',help = "The path to the coarse annotations. File format should be `.sav`")
ap.add_argument('model_dir',help = "Where to load the model (to conduct feature extraction)")
ap.add_argument('feature_save_root',help = "Where the mapped features will be saved")
ap.add_argument('knn_save_root',help = "Where the KNN results (distance and index) will be saved")
ap.add_argument('heatmap_save_root',help = "Where the predicted heatmap will be saved")
ap.add_argument("--remove_blank", type = int, help = "How to remove blank regions (i.e. identify tissue regions) of WSI. We provide three functions: 0: convert to HSV and then OTSU threshold on H and S channel; 1: apply [235, 210, 235] on RGB channel, respecitively; 2: convert to gray image, then OTSU threshold.  Default is 0. For new dataset, the user is encouraged to write customed function",default = 0)
ap.add_argument("--focal_loss", type = str2bool, help = "Whether or not to use focal loss (True: using focal loss; Flase: using cross entropy), default is CE", default=False)
ap.add_argument("--patch_shape",type = int, help = "Patch shape(size), default is 256", default=256)
ap.add_argument("--unit",type = int, help = "Samllest unit when cropping patches, default is 256", default=256)
ap.add_argument("--gpu", help = "gpu",default = '0')
args = ap.parse_args()    
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu   
loader_kwards = {'num_workers':1, 'pin_memory':True} 

print('-------------------------Read slide-------------------------------')
slide_ob = openslide.OpenSlide(args.slide_root+'/'+args.slide_ID+args.slide_format)

print('-------------------------Read coarse annotations-------------------------------')
with open(args.ca_path,'rb') as handle:
    CA = pickle.load(handle)
handle.close()
annot_coarse_inner = CA[args.slide_ID]['annotations_inner']
annotations_outer = CA[args.slide_ID]['annotations_outer']

print('-------------------------Create masks (tissue mask and label mask)------------------------------')   
# Maks are all loww-resolution versions, downsampled by args.unit
boundary_ratio = [0,0,0,0] # [a,b,c,d] means cropping a * height, b * height, c * width, d * width from the top, bottom, left and right sides of the WSI. It can be useful if their are some smear at the boarder of slides
mask_coarse = np.array(crop_boundary(create_tumor_mask(slide_ob,[annotations_outer,annot_coarse_inner],args.unit),boundary_ratio),dtype=np.uint8) # genertae label mask based on coarse annoations
thumbnail = np.array(slide_ob.get_thumbnail((mask_coarse.shape[1],mask_coarse.shape[0])))[:,:,:3] # thumbnail 
if args.remove_blank == 0:
    mask_tissue = crop_boundary(HSV_otsu(thumbnail),boundary_ratio)
elif args.remove_blank == 1:
    mask_tissue = crop_boundary(binary_PAIP(thumbnail),boundary_ratio)
else:
    mask_tissue = crop_boundary(GRAY_otsu(thumbnail,True),boundary_ratio)
mask_tissue = cv.resize(mask_tissue,(mask_coarse.shape[1],mask_coarse.shape[0])) # generate tissue mask to exclude blank regions in the WSI

print("----------------------------------Dataset-----------------------------------------------")
ROW, COL = np.where(mask_tissue==1)
dat_set = TSpatch_apply(slide_dir = args.slide_root + '/'+ args.slide_ID + args.slide_format, 
                              unit=args.unit,
                              patch_shape = args.patch_shape,
                              ROW = ROW, 
                              COL = COL, 
                              transform = transforms.Compose([
                                  transforms.Resize(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                             )                 
dat_loader = data_utils.DataLoader(dat_set, batch_size = 1, shuffle = False)

print("----------------------------------Load in model-----------------------------------------------")
vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
vgg.classifier[6] = Identity()
model = vgg_binary(vgg)   
model.load_state_dict(torch.load(args.model_dir))
model.cuda()

print('--------------------------------Applying model to map patches to the feature space (R^4096)------------------------------------------')
Feature = np.zeros((len(ROW),4096)) # initialize a matrix to save extracted features             
for batch_idx, (patch,row,col) in enumerate(dat_loader):
    patch = patch.cuda()
                   
    if batch_idx % 100 == 0:
        print(batch_idx ,'/', len(ROW))
    feature = model.feature_extractor(patch)
    Feature[batch_idx,:] = feature.data.cpu().numpy()                 
Feature = Feature.astype('float32')                  
np.save(args.feature_save_root+'/feature_'+args.slide_ID+'.npy',Feature)
                

print('------------------------------------Faiss: A fast implementation of KNN------------------------------------------')   
# For Faiss, please refer to: https://github.com/facebookresearch/faiss
X =  Feature
X_normalize = normalize(X)
index = faiss.IndexFlatL2(4096)
index.add(X_normalize) 
distances, indices = index.search(X_normalize, k=5)
knn = {'distance':distances,
       'indices':indices}
with open(args.knn_save_root+'/knn_'+args.slide_ID+'.sav','wb') as handle:
    pickle.dump(knn,handle,protocol = pickle.HIGHEST_PROTOCOL)
handle.close()

print('---------------------------------Generate a binary heatmap ---------------------------------')
# Every single patch will be relabeled using the KNN classifier
mask_coarse_flatten = np.array([1 if mask_coarse[ROW[i],COL[i]]==1 else 0 for i in range(len(ROW))]) # faltten the label mask
Vote = mask_coarse_flatten[indices][:,:5]
Vote_result = np.array(np.mean(Vote,axis=1)>0.5,np.uint8)
predicted_positive = np.where(Vote_result == 1)
predicted_negative = np.where(Vote_result == 0)
heatmap_5 = np.zeros_like(mask_tissue,dtype=float)
heatmap_5[ROW[predicted_positive],COL[predicted_positive]] = 1
heatmap_5[mask_tissue!=1] = np.nan
np.save(args.heatmap_save_root+'/heatmap_'+args.slide_ID+'.npy',heatmap_5)
