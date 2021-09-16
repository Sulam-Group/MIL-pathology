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
import pickle

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

# MIL bag construction
class TSbags_train_random(data_utils.Dataset):
    def __init__(self, slide_dir, unit, patch_shape, ROW_tumor, COL_tumor, ROW_stroma, COL_stroma, num_bag_tumor,num_bag_stroma,length_bag_mean, transform):
        self.slide = openslide.OpenSlide(slide_dir)
        self.width, self.height = self.slide.dimensions
        self.unit = unit
        self.patch_shape = patch_shape
        self.ROW_tumor = ROW_tumor
        self.COL_tumor = COL_tumor
        self.ROW_stroma = ROW_stroma
        self.COL_stroma = COL_stroma
        self.num_bag_tumor = num_bag_tumor
        self.num_bag_stroma = num_bag_stroma
        self.length_bag_mean = length_bag_mean
        self.transform = transform
        self.bags_list, self.labels_list = self._create_bags()  
    def _create_bags(self):            
        bags_list = []
        labels_list = []
        while len(bags_list)< self.num_bag_tumor:
            length_bag = binom.rvs (n=int(self.length_bag_mean*2), p=0.5)
            indices = np.random.randint(0,len(self.ROW_tumor),length_bag)
            bags_list.append((self.ROW_tumor[indices], self.COL_tumor[indices]))
            labels_list.append(1)
        while len(bags_list)< self.num_bag_tumor+self.num_bag_stroma:
            length_bag = binom.rvs (n=int(self.length_bag_mean*2), p=0.5)
            indices = np.random.randint(0,len(self.ROW_stroma),length_bag)
            bags_list.append((self.ROW_stroma[indices], self.COL_stroma[indices]))
            labels_list.append(0)
        return bags_list, labels_list
    def _pack_one_bag(self,row_list, col_list):
        Bag = []
        for i in range(len(row_list)):
            row_unit, col_unit = row_list[i], col_list[i]
            upperLeft_x = int(col_unit * self.unit + self.unit/2 - self.patch_shape/2)
            upperLeft_y = int(row_unit * self.unit + self.unit/2 - self.patch_shape/2)
#             if upperLeft_x<0 or upperLeft_y<0 or upperLeft_x+self.patch_shape>self.width or upperLeft_y+self.patch_shape>self.height:
#                 continue            
#             print((upperLeft_x, upperLeft_y),0,(self.patch_shape,self.patch_shape))
            patch = self.slide.read_region((upperLeft_x, upperLeft_y),0,(self.patch_shape,self.patch_shape))
            patch = Image.fromarray(np.array(patch)[:,:,:3])
            if self.transform is not None:
                patch = self.transform(patch)
            Bag.append(patch)
        Bag = np.stack(Bag,axis=0)
        return Bag  
    def __len__(self):
        return len(self.bags_list)  
    def __getitem__(self, index):
        row_list, col_list = self.bags_list[index]
        bag = self._pack_one_bag(row_list, col_list)
        label = self.labels_list[index]
        return bag, label

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

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument('slide_root',help = "The location of the whole-slide image.")
ap.add_argument('slide_ID',help = "Name of the whole-slide image.")
ap.add_argument('slide_format',help = "Dataformat the whole-slide image. Permitted format can be `.svs`, `,ndpi`, and `.tif`.")
ap.add_argument('ca_path',help = "The path to the coarse annotations. File format should be `.sav`")
ap.add_argument('model_save_root',help = "Where model will be saved")
ap.add_argument("--remove_blank", type = int, help = "How to remove blank regions (i.e. identify tissue regions) of WSI. We provide three functions: 0: convert to HSV and then OTSU threshold on H and S channel; 1: apply [235, 210, 235] on RGB channel, respecitively; 2: convert to gray image, then OTSU threshold.  Default is 0. For new dataset, the user is encouraged to write customed function",default = 0)
ap.add_argument("--length_bag_mean",type = int, help = "Average length of bag (Binomial distribution),default = 10",default = 10)
ap.add_argument("--num_bags",type = int, help = "Number of bags to train,default = 1000", default = 1000)
ap.add_argument("--focal_loss", type = str2bool, help = "Whether or not to use focal loss (True: using focal loss; Flase: using cross entropy), default is FL", default=True)
ap.add_argument("--patch_shape",type = int, help = "Patch shape(size), default is 256", default=256)
ap.add_argument("--unit",type = int, help = "Samllest unit when cropping patches, default is 256", default=256)
ap.add_argument("--gpu", help = "gpu",default = '0')
ap.add_argument("--lr",type = float, help = "Initial Learning rate, default is 0.00005", default=0.00005)
ap.add_argument("--step_size",type = int, help = "Step size when decay learning rate, default is 1", default=1)
ap.add_argument("--reg",type = float, help = "Reg,default is 10e-5", default=10e-5)
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

print('-------------------------Identify the tissue patches ------------------------------')   
ROW_tumor, COL_tumor = np.where((mask_coarse==1) & (mask_tissue==1))
ROW_stroma, COL_stroma = np.where((mask_coarse==0) & (mask_tissue==1))
print("There are {} tumor patches (coarsely assigned), and {} non-tumor patches (coarsely assigned).".format(len(ROW_tumor),len(ROW_stroma)))

print('----------------------------------Training preparation----------------------------------------')
feature_extractor = 'vgg'
model = Attention_modern(load_vgg16(),args.focal_loss) # model architecture: MIL framework with a VGG encoder
model.cuda() # send model to GPU
optimizer = optim.Adam(model.parameters(),lr=args.lr, betas=(0.9, 0.999), weight_decay =args.reg) # define optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = 0.5) # define learning schedule: learning rate decay to 50% every 100 bags

print('----------------------------------Training ----------------------------------------')
Train_accuracy = [] # list to record the change of accuracy during training
Train_loss = [] # list to record the change of loss during training

# Instead of create the dataset pnce before training, we pack 100 bags (50 positive, 50 negative) at each time, training, and then repeate this process
for epoch in range(1,int(args.num_bags/100)+1):  
    dat_set = TSbags_train_random(slide_dir = args.slide_root + '/' + args.slide_ID + args.slide_format, 
                                  unit=args.unit,
                                  patch_shape = args.patch_shape,
                                  ROW_tumor = ROW_tumor, 
                                  COL_tumor = COL_tumor, 
                                  ROW_stroma = ROW_stroma, 
                                  COL_stroma = COL_stroma, 
                                  num_bag_tumor = 50,
                                  num_bag_stroma = 50,
                                  length_bag_mean = args.length_bag_mean, 
                                  transform = transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                 )      
    dat_loader = data_utils.DataLoader(dat_set, batch_size = 1, shuffle = True)
    
    train_accuracy, train_loss = train(dat_loader, model, optimizer, scheduler)
    print("\nepoch = {}, accuracy in training set is {}, loss in train set is {}".format(epoch, train_accuracy, train_loss))
    Train_accuracy.append(train_accuracy)
    Train_loss.append(train_loss)
    scheduler.step()

print('-------------------------Saving model-------------------------------')
torch.save(model.state_dict(), args.model_save_root+'/model_'+args.slide_ID+'.pth')
