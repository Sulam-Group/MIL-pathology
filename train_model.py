# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:18:58 2020

@author: Jasmine
"""
import os
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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

"""
Dataset:
    1) Pathology_Patches: used to load on patches
    2) Pathology_Bags: used to pack into bags
    3) prepare_data: used to seperate training set and validation set
"""
class Pathology_Patches(data_utils.Dataset):
    def __init__(self,root_dir_list, patch_shape, tissue, transform):
        self.root_dir_list = root_dir_list
        self.patch_shape = patch_shape
        self.tissue = tissue
        self.transform = transform
        self.path_list = self.get_path()

    def get_path(self):
        path_list = []
        for root_dir in self.root_dir_list:
            sub_dir = os.path.join(root_dir,str(self.patch_shape[0]))
            if self.tissue == 'tumor':
                sub_patch_dir = os.path.join(sub_dir,'tumor/non-blank')
            elif self.tissue == 'stroma':
                sub_patch_dir = os.path.join(sub_dir,'stroma/non-blank')
            else:
                print("Invalid tissue type")
            for path in os.listdir(sub_patch_dir):
                full_path = os.path.join(sub_patch_dir,path)
                if os.path.isfile(full_path):
                    path_list.append(full_path)
        return path_list
    def __len__(self):
        return len(self.path_list)
    def __getitem__(self,index):
        img = imageio.imread(self.path_list[index])
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    
class Pathology_Bags(data_utils.Dataset):
    def __init__(self, tumor_loader, stroma_loader, num_bag, length_bag, seed):
        self.tumor_loader = tumor_loader
        self.stroma_loader = stroma_loader
        self.num_bag = num_bag
        self.length_bag = length_bag
        self.r = np.random.RandomState(seed)
        self.bags_list, self.labels_list = self._create_bags()
        
    
    def _create_bags(self):
        for patch_idx, patch in enumerate(self.tumor_loader):
            all_tumor_imgs = patch
            #print(all_tumor_imgs.shape)
        for patch_idx, patch in enumerate(self.stroma_loader):
            all_stroma_imgs = patch
            
        bags_list = []
        labels_list = []
        while (len(bags_list)< int(self.num_bag/2)):
            print(len(bags_list))
            indices = torch.LongTensor(self.r.randint(0, all_tumor_imgs.shape[0], self.length_bag))
            #print(indices)
            bags_list.append(all_tumor_imgs[indices])
            labels_list.append(1)
        while (len(bags_list)< self.num_bag):
            print(len(bags_list))
            indices = torch.LongTensor(self.r.randint(0, all_stroma_imgs.shape[0], self.length_bag))
            #print(indices)
            bags_list.append(all_stroma_imgs[indices])
            labels_list.append(0)
        return bags_list, labels_list
    
    def __len__(self):
        return len(self.bags_list)
    
    def __getitem__(self, index):
        bag = self.bags_list[index]
        label = self.labels_list[index]
        return bag, label
        
    
import random
def prepare_data(args):
    trainval_tumor_set = Pathology_Patches( root_dir_list = args.root_dir_list,
                               patch_shape = args.patch_shape, 
                               tissue = 'tumor',
                               transform = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.777, 0.778, 0.769),
                                                                                  (0.124,0.128,0.128))])
                              )
    trainval_stroma_set = Pathology_Patches( root_dir_list = args.root_dir_list,
                               patch_shape = args.patch_shape, 
                               tissue = 'stroma',
                               transform = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.777, 0.778, 0.769),
                                                                                  (0.124,0.128,0.128))])
                              )


    train_tumor_index = random.sample(range(len(trainval_tumor_set)),k = int(len(trainval_tumor_set)*0.8))
    val_tumor_index = [i for i in range(len(trainval_tumor_set)) if i not in train_tumor_index]
    train_stroma_index = random.sample(range(len(trainval_stroma_set)),k = int(len(trainval_stroma_set)*0.8))
    val_stroma_index = [i for i in range(len(trainval_stroma_set)) if i not in train_stroma_index]

    train_tumor_set = data_utils.Subset(trainval_tumor_set, train_tumor_index )
    val_tumor_set = data_utils.Subset(trainval_tumor_set, val_tumor_index)
    train_stroma_set = data_utils.Subset(trainval_stroma_set, train_stroma_index )
    val_stroma_set = data_utils.Subset(trainval_stroma_set, val_stroma_index)
    
    print("Tumor patches in training set:", len(train_tumor_set))
    print("Stroma patches in training set:",len(train_stroma_set))
    print("Tumor patches in validation set:",len(val_tumor_set))
    print("Stroma patches in validation set:",len(val_stroma_set))
    
    Train_tumor_loader_list = []
    for i in range(1,5):
        fold_size = int(len(train_tumor_set)/4)  
        if i == 4:
            train_tumor_set_fold = data_utils.Subset(train_tumor_set, range(int((i-1)*fold_size),len(train_tumor_set)))
        else:
            train_tumor_set_fold = data_utils.Subset(train_tumor_set, range(int((i-1)*fold_size),int(i*fold_size)))
        print("Fold {}, Tumor patches in sub-training set:{}".format(i,len(train_tumor_set_fold)))
        train_tumor_loader_fold = data_utils.DataLoader(train_tumor_set_fold, batch_size = len(train_tumor_set_fold), 
                                                        shuffle = False)
        Train_tumor_loader_list.append(train_tumor_loader_fold)
                                                        
    Train_stroma_loader_list = []
    for i in range(1,5):
        fold_size = int(len(train_stroma_set)/4)  
        if i == 4:
            train_stroma_set_fold = data_utils.Subset(train_stroma_set, range(int((i-1)*fold_size),len(train_stroma_set)))
        else:
            train_stroma_set_fold = data_utils.Subset(train_stroma_set, range(int((i-1)*fold_size),int(i*fold_size)))
        print("Fold {}, Stroma patches in sub-training set:{}".format(i,len(train_stroma_set_fold)))
        train_stroma_loader_fold = data_utils.DataLoader(train_stroma_set_fold, batch_size = len(train_stroma_set_fold), 
                                                        shuffle = False)
        Train_stroma_loader_list.append(train_stroma_loader_fold)
    
    val_tumor_loader = data_utils.DataLoader(val_tumor_set, batch_size = len(val_tumor_set), shuffle = False)
    val_stroma_loader = data_utils.DataLoader(val_stroma_set, batch_size = len(val_stroma_set), shuffle = False)
    
    return Train_tumor_loader_list, val_tumor_loader, Train_stroma_loader_list, val_stroma_loader

"""
Model:
    1) Attention_92: architecture of Attention-MIL model
    2) 
"""
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
        A = F.softmax(A,dim=1)

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


def train(epoch, loader):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(loader):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        optimizer.zero_grad()
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        error, _, _ =model.calculate_classification_error(data, bag_label)
        train_error += error
        loss.backward()
        optimizer.step()

    train_loss /= len(loader)
    train_error /= len(loader)
    print('Epoch: {}, Loss: {:.4f}, Train error : {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))
    return 1 - train_error, train_loss.cpu().numpy()[0]

def test_coarse(loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
  
    for batch_idx, (data, label) in enumerate(loader):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.cpu().data.numpy()[0]
        error, predicted_label, Y_prob = model.calculate_classification_error(data, bag_label)
        test_error += error
    
    test_error /= len(loader)
    test_loss /= len(loader)
    
    accuracy = 1 - test_error
    return accuracy, test_loss


def test_detail(loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    Y_probs = []
    Labels = []
    Attention_weights_positive_bags = []
    Attention_weights_negative_bags = []
    Attention_weights_positive_hat_bags = []
    Attention_weights_negative_hat_bags = []
    for batch_idx, (data, label) in enumerate(loader):
        bag_label = label
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.cpu().data.numpy()[0]
        error, predicted_label, Y_prob = model.calculate_classification_error(data, bag_label)
        test_error += error
        Y_probs.append(Y_prob.cpu().data.numpy()[0][0])
        Labels.append(label)
        if label.numpy()[0]==1:
            Attention_weights_positive_bags.extend(attention_weights.cpu().data.numpy()[0].tolist())
        else:
            Attention_weights_negative_bags.extend(attention_weights.cpu().data.numpy()[0].tolist())
            
        if predicted_label.cpu().numpy()[0]==1:
            Attention_weights_positive_hat_bags.extend(attention_weights.cpu().data.numpy()[0].tolist())
        else:
            Attention_weights_negative_hat_bags.extend(attention_weights.cpu().data.numpy()[0].tolist())
    
    test_error /= len(loader)
    test_loss /= len(loader)
    
    accuracy = 1 - test_error
    bag_level = (Y_probs, Labels)
    instance_level = {
        'actual positive':Attention_weights_positive_bags,
        'actual negative':Attention_weights_negative_bags,
        'predicted positive':Attention_weights_positive_hat_bags,
        'predicted negative':Attention_weights_negative_hat_bags,
        
    }
    return accuracy, test_loss, bag_level, instance_level



"""
Specify arguments here
"""
class Args:
    def __init__(self):
        self.root_dir_list=['D:/Ashel-slide/458603']
        self.patch_shape = (92,92)
        self.model = Attention_92()
        self.num_bag_train = 100
        self.num_bag_val = 500
        self.length_bag = 100
        self.seed = 1   
        self.epochs = 20
        self.lr = 0.0001
        self.reg = 10e-5
args = Args()
loader_kwards = {'num_workers':1, 'pin_memory':True} 
torch.cuda.manual_seed(args.seed)



"""Load in data
"""
Train_tumor_loader_list, val_tumor_loader, Train_stroma_loader_list, val_stroma_loader = prepare_data(args)
print("\nLoad in validation set")
val_set = Pathology_Bags(tumor_loader = val_tumor_loader, 
                           stroma_loader = val_stroma_loader, 
                           num_bag = args.num_bag_val, 
                           length_bag = args.length_bag, 
                           seed = args.seed)
val_loader = data_utils.DataLoader(val_set, batch_size = 1, shuffle = False)
len_negative_bag = 0
len_positive_bag = 0
for bag_idx, (bag, label) in enumerate(val_loader):
    print(bag.shape)
    if label.numpy()[0] == 0:
        len_negative_bag += 1
    else:
        len_positive_bag += 1
print("There are {} positive bags, {} negative bags in validation set".format(len_positive_bag, len_negative_bag))

"""
Initiliaze model
"""
print("\nInitialize model")
model = args.model
model.cuda()
optimizer = optim.Adam(model.parameters(),lr=args.lr, betas=(0.9, 0.999), weight_decay =args.reg)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)

"""
Start training .....
"""
print("\nStart training")
Train_accuracy = []
Val_accuracy = []
Loss_train = []
Loss_val = []
Epochs = np.arange(1,args.epochs+1).tolist()
for epoch in range(1,args.epochs+1):  
    train_set = Pathology_Bags(tumor_loader = Train_tumor_loader_list[epoch % 4], 
                           stroma_loader = Train_stroma_loader_list[epoch % 4], 
                           num_bag = args.num_bag_train, 
                           length_bag = args.length_bag, 
                           seed = args.seed)

    train_loader = data_utils.DataLoader(train_set, batch_size = 1, shuffle = True)
    
    train_accuracy, loss_train = train(epoch, train_loader)
    val_accuracy, loss_val = test_coarse(val_loader)
    
    print("epoch = {}, accuracy in training set is {}, accuracy in valiation set is {}".format(epoch, train_accuracy, val_accuracy))
    print("epoch = {}, loss in training set is {}, loss in valiation set is {}".format(epoch, loss_train, loss_val))
    Train_accuracy.append(train_accuracy)
    Val_accuracy.append(val_accuracy)
    Loss_train.append(loss_train)
    Loss_val.append(loss_val)
    scheduler.step()

torch.save(model.state_dict(), os.path.join(args.root_dir_list[0], "model_92_100.pth"))

"""
Figures for monitoring the training and evaluate the final result
"""
plt.figure()
plt.plot(Epochs, Train_accuracy, label = "Accuracy in training set")
plt.plot(Epochs, Val_accuracy, label = "Accuracy in validatiohn set")
plt.title("Train on {} bags, test on {} bags, {} images in each bag, image size = {}".format(int(args.num_bag_train),
                                                                                             int(args.num_bag_val),
                                                                                               args.length_bag,
                                                                                               args.patch_shape))
plt.ylim([0.0,1.05])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(os.path.join(args.root_dir_list[0],"accuracy.png"))

plt.figure()
plt.plot(Epochs, Loss_train, label = "Loss in training set")
plt.plot(Epochs, Loss_val, label = "Loss in validation set")
plt.title("Train on {} bags, test on {} bags, {} images in each bag, image size = {}".format(int(args.num_bag_train),
                                                                                             int(args.num_bag_val),
                                                                                               args.length_bag,
                                                                                               args.patch_shape))

plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(args.root_dir_list[0],"loss.png"))

train_accuracy, train_loss, (train_prob, train_labels), _ = test_detail(train_loader)
val_accuracy, val_loss, (val_prob, val_labels), _ = test_detail(val_loader)
train_auc = roc_auc_score(train_labels, train_prob)
val_auc = roc_auc_score(val_labels, val_prob)
fpr_train, tpr_train, _ = roc_curve(train_labels, train_prob)
fpr_val, tpr_val, _ = roc_curve(val_labels, val_prob)
plt.figure()
plt.plot(fpr_train, tpr_train, label = "ROC on training set (AUC=%.2f)"%train_auc)
plt.plot(fpr_val, tpr_val, label = "ROC on validation set (AUC=%.2f)"%val_auc)
plt.title("Train on {} bags, test on {} bags, {} images in each bag, image size = {}".format(int(args.num_bag_train),
                                                                                             int(args.num_bag_val),
                                                                                               args.length_bag,
                                                                                           args.patch_shape))
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.legend()
plt.savefig(os.path.join(args.root_dir_list[0],"roc.png"))


"""
Instance level analysis
"""
_, _, (Y_probs, Labels), instance_level = test_detail(val_loader)
fig, (ax1, ax2) = plt.subplots(1,2, figsize =(10,5))
ax1.boxplot(instance_level['actual positive'])
ax1.set_title("Weights of instances in actual positive bags")
ax2.boxplot(instance_level['actual negative'])
ax2.set_title("Weights of instances in actual negative bags")

fig, (ax1, ax2) = plt.subplots(1,2, figsize =(10,5))
ax1.boxplot(instance_level['predicted positive'])
ax1.set_title("Weights of instances in predicted positive bags")
ax1.set_ylim(0,1)
ax2.boxplot(instance_level['predicted negative'])
ax2.set_title("Weights of instances in predicted negative bags")
ax2.set_ylim(0,1)
plt.savefig(os.path.join(args.root_dir_list[0],"boxplot.png"))
print("There are {} instances in actual positive bags, {} high-weighted instances".format(len(instance_level['actual positive']),
                                                                                        len([i for i in instance_level['actual positive'] if i>0.5 ]) ))
print("There are {} instances in predicted positive bags, {} high-weighted instances".format(len(instance_level['predicted positive']),
     
                                                                                             len([i for i in instance_level['predicted positive'] if i>0.5 ]) ))

print("Standard deviation of weights in predicted positive bags are %.2f"% np.std(instance_level['predicted positive']))
print("Standard deviation of weights in predicted negative bags are %.2f"% np.std(instance_level['predicted negative']))
print("Standard deviation of weights in actual positive bags are %.2f"% np.std(instance_level['actual positive']))
print("Standard deviation of weights in actual negative bags are %.2f"% np.std(instance_level['actual negative']))

import seaborn as sns
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
sns.distplot(instance_level['predicted positive'], label = "predicted positive")
sns.distplot(instance_level['predicted negative'], label = "predicted negative")
plt.legend()
plt.title("Distribution of weights(bags grouped by prediction)")

plt.subplot(1,2,2)
sns.distplot(instance_level['actual positive'], label = "actual positive")
sns.distplot(instance_level['actual negative'], label = "actual negative")
plt.legend()
plt.title("Distribution of weights(bags grouped by ground truth)")

plt.savefig(os.path.join(args.root_dir_list[0],"distribution_weights.png"))


"""
Extracting typical patches and save them
"""
High_weighted_imgs_predicted_positive = []
High_weights_predicted_positive = []
Low_weighted_imgs_predicted_positive = []
Low_weights_predicted_positive = []
Bag_prob_predicted_positive = []

High_weighted_imgs_predicted_negative = []
High_weights_predicted_negative = []
Low_weighted_imgs_predicted_negative = []
Low_weights_predicted_negative = []
Bag_prob_predicted_negative = []

for batch_idx, (data, label) in enumerate(val_loader):
    bag_label = label
    data, bag_label = data.cuda(), bag_label.cuda()
    data, bag_label = Variable(data), Variable(bag_label)
    _, attention_weights = model.calculate_objective(data, bag_label)
    _, predicted_label, Y_prob = model.calculate_classification_error(data, bag_label)
    attention_weights = attention_weights.cpu().data.numpy()[0].tolist()
    predicted_label = predicted_label.cpu().numpy()[0][0]
    Y_prob = Y_prob.cpu().data.numpy()[0][0]
    label = label.numpy()[0]
    data = data.cpu().detach().numpy()
    index_highest_weight = np.argmax(attention_weights)
    img_highest_weight = data[0,index_highest_weight,:,:,:].transpose((1,2,0))
    img_highest_weight = (img_highest_weight - np.min(img_highest_weight))/(np.max(img_highest_weight) - np.min(img_highest_weight))
    
    index_lowest_weight = np.argmin(attention_weights)
    img_lowest_weight = data[0,index_lowest_weight,:,:,:].transpose((1,2,0))
    img_lowest_weight = (img_lowest_weight - np.min(img_lowest_weight))/(np.max(img_lowest_weight) - np.min(img_lowest_weight))
    if predicted_label == 1:
        High_weighted_imgs_predicted_positive.append(img_highest_weight)
        High_weights_predicted_positive.append(attention_weights[index_highest_weight])
        Low_weighted_imgs_predicted_positive.append(img_lowest_weight)
        Low_weights_predicted_positive.append(attention_weights[index_lowest_weight])        
        #print(attention_weights[index_highest_weight])
        Bag_prob_predicted_positive.append(Y_prob)
    else:
        High_weighted_imgs_predicted_negative.append(img_highest_weight)
        High_weights_predicted_negative.append(attention_weights[index_highest_weight])
        Low_weighted_imgs_predicted_negative.append(img_lowest_weight)
        Low_weights_predicted_negative.append(attention_weights[index_lowest_weight]) 
        Bag_prob_predicted_negative.append(Y_prob)


f,a = plt.subplots(10,5, squeeze = False, figsize = (25,50))
for row in range(10):
    for col in range(5):
        index = row*5+col
        print(index)
        img = High_weighted_imgs_predicted_positive[index]
        weight = High_weights_predicted_positive[index]
        a[row,col].imshow(img)
        a[row,col].set_title("weight = %.2f"%weight)
f.tight_layout()
plt.savefig(os.path.join(args.root_dir_list[0],"group1.png"))

f,a = plt.subplots(10,5, squeeze = False, figsize = (25,50))
for row in range(10):
    for col in range(5):
        print(index)
        index = row*5+col
        img = High_weighted_imgs_predicted_negative[index]
        weight = High_weights_predicted_negative[index]
        a[row,col].imshow(img)
        a[row,col].set_title("weight = %.2f"%weight)
f.tight_layout()
plt.savefig(os.path.join(args.root_dir_list[0],"group2.png"))

f,a = plt.subplots(10,5, squeeze = False, figsize = (25,50))
for row in range(10):
    for col in range(5):
        index = row*5+col
        img = Low_weighted_imgs_predicted_positive[index]
        weight = Low_weights_predicted_positive[index]
        a[row,col].imshow(img)
        a[row,col].set_title("weight = %.2f"%weight)
f.tight_layout()
plt.savefig(os.path.join(args.root_dir_list[0],"group3.png"))

f,a = plt.subplots(10,5, squeeze = False, figsize = (25,50))
for row in range(10):
    for col in range(5):
        index = row*5+col
        img = Low_weighted_imgs_predicted_negative[index]
        weight = Low_weights_predicted_negative[index]
        a[row,col].imshow(img)
        a[row,col].set_title("weight = %.2f"%weight)
f.tight_layout()
plt.savefig(os.path.join(args.root_dir_list[0],"group4.png"))


    