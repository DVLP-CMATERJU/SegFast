# -*- coding: utf-8 -*-
from models import fcn,segnet,pspnet,segfast,segfast_mobile,unet,segfast_basic,segfast_v2
from torch.utils.data import Dataset
import torch,os
from PIL import Image
import torchvision.transforms as t
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import csv
from torchvision.utils import save_image

def load_model(model_name,noc):
    if model_name=='fcn':
        model = fcn.FCN8s(noc)
    if model_name=='segnet':
        model = segnet.SegNet(3,noc)
    if model_name=='pspnet':
        model = pspnet.PSPNet(noc)
    if model_name=='unet':
        model = unet.UNet(noc)
    if model_name == 'segfast':
        model =  segfast.SegFast(64,noc)
    if model_name == 'segfast_basic':
        model =  segfast_basic.SegFast_Basic(64,noc)
    if model_name == 'segfast_mobile':
        model =  segfast_mobile.SegFast_Mobile(noc)
    if model_name == 'segfast_v2_3':
        model =  segfast_v2.SegFast_V2(64,noc,3)
    if model_name == 'segfast_v2_5':
        model =  segfast_v2.SegFast_V2(64,noc,5)
    return model

class getDataset(Dataset):
    def __init__(self, image_path, label_path, image_transform=None, label_transform=None, size=None, num_of_classes=2):
        self.images = sorted(os.listdir(image_path))
        self.labels = sorted(os.listdir(label_path))
        self.noc = num_of_classes-1
        assert(len(self.images)==len(self.labels)), 'The two folders do not have same number of images'
        for i,filename in enumerate(self.images):
            self.images[i] = image_path+'/'+self.images[i]
            self.labels[i] = label_path+'/'+self.labels[i]
        if size==None:
            self.size=(224,224)
        else:
            self.size=size

        if image_transform==None:
            self.image_transform = t.Compose([t.Resize(self.size),t.ToTensor()])
        else:
            self.image_transform = image_transform

        if label_transform==None:
            self.label_transform = t.Compose([t.Resize(self.size,interpolation=Image.NEAREST)])
        else:
            self.label_transform = label_transform
    def __getitem__(self, index):
        image = self.image_transform(Image.open(self.images[index]))
        if image.size()[0]==1:
            print("1D image")
            image = torch.cat([image]*3)
        t=Image.open(self.labels[index])
        label = self.label_transform(t)
        label=np.array(label)
        s = label.shape
        if(len(s) == 3):
        		if(s[2]==3):
        		    #print("e")
        		    label = label[:,:,0]
        label=torch.from_numpy(label).long()
        return (image,label.squeeze())
    def __len__(self):
        return (len(self.images))

def get_args():
    parser = argparse.ArgumentParser(description='''Encoder Decoder
                                     Architecture with skip connections
                                     for Image segmentation''')
    parser.add_argument('--model', default = 'segnet',
                        help='segfast|unet|segnet|segfast_mobile|segfast_basic|pspnet|fcn|')
    parser.add_argument('--dataset_path', default = None,
                        help='choose dataset folder path')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='max number of training epochs. default=10')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='training batch size. default=4')
    parser.add_argument('--num_of_workers', type=int, default=4,
                        help='number of cpu threads for data loading. default=4')
    parser.add_argument('--save_path', default='./',
                        help='''path to save output files''')
    parser.add_argument('--fresh_train', default= 1,
                        help='''1 for fresh training, 0 for loading model''')

    return parser.parse_args()

def get_data_path(dataset,config_file):
    path=dict([])
    with open (config_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            k,v,noc = line.split(',')
            path[k] = [v,int(noc)]
        return path[dataset]
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calc_pixel_level_accuracy(a,b):
    acc = (a==b).sum()/(a.shape[0]*a.shape[1]*a.shape[2])
    return acc

def calc_conf(a,b,noc,c_m): # a = actual, b = predicted
    a = a.view(-1)
    b = b.view(-1)
    a = a.data
    b = b.data
#    if type(a) == torch.autograd.Variable:
#        a = a.data
#    if type(b) == torch.autograd.Variable:
#        b = b.data
    for i in range(a.shape[0]):
        c_m[a[i],b[i]]+=1
    return c_m

def calc_measures(noc,c_m):
    intersection = np.array([c_m[i,i] for i in range(noc)])
    union = np.array([c_m[i,:].sum()+c_m[:,i].sum()-c_m[i,i] for i in range(noc)])
    iou = (1+intersection)/(1+union)
    precision = np.array([(1+c_m[i,i])/(1+c_m[:,i].sum()) for i in range(noc)])
    recall = np.array([(1+c_m[i,i])/(1+c_m[i,:].sum()) for i in range(noc)])
    f_1_score = (1+(2*intersection))/(1+np.array([c_m[i,:].sum()+c_m[:,i].sum() for i in range(noc)]))
    acc = sum([c_m[i,i] for i in range(noc)])/c_m.sum()
    return iou,precision,recall,f_1_score,acc

class Trainer():
    def __init__(self,model,train_loader,val_loader,save_path,epochs,noc):
        self.noc = noc
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.model = model
        self.noe = epochs
    def train(self):
        loss_fn = nn.NLLLoss(size_average=True)
        optimizer = optim.Adam(self.model.parameters())
        train_loss_vs_epoch = []
        val_loss_vs_epoch = []
        mkdir(self.save_path)
        print('TRAINING STARTED')
        counter = 0
        best_val_loss = 99999
        for epoch in range(self.noe):
            train_loss = 0
            self.model = self.model.train(True)
            for input,label in self.train_loader:
                optimizer.zero_grad()
                input,label = input.cuda(),label.cuda()
                output = self.model(input)
                if (type(output) == tuple):
                    loss = loss_fn(F.log_softmax(output[0],dim=1),label) + loss_fn(F.log_softmax(output[1],dim=1),label)
                else:
                    loss = loss_fn(F.log_softmax(output,dim=1),label)
                train_loss+=loss.item()
                loss.backward()
                optimizer.step()
            train_loss = train_loss/float(len(self.train_loader))
            train_loss_vs_epoch.append([train_loss])
            # SAVE IMAGES
            mkdir(self.save_path+'/during_training/images')
            mkdir(self.save_path+'/during_training/labels')
            mkdir(self.save_path+'/during_training/predicted')
            mkdir(self.save_path+'/during_training/uncertainties')
            output_classes = torch.max(output,dim=1)[1]
            uncertainties = torch.max(F.softmax(output,dim=1),dim=1)[0]
            save_image(input[0],self.save_path+'/during_training/images/%03d.png'%(epoch))
            save_image((label.float()/float(self.noc))[0],self.save_path+'/during_training/labels/%03d.png'%(epoch))
            save_image((output_classes.float()/float(self.noc))[0],self.save_path+'/during_training/predicted/%03d.png'%(epoch))
            save_image(uncertainties[0],self.save_path+'/during_training/uncertainties/%03d.png'%(epoch))
            val_loss = 0
            self.model = self.model.train(False)
            for input,label in self.val_loader:
                input,label = input.cuda(),label.cuda()
                output = self.model(input)
                loss = loss_fn(F.log_softmax(output,dim=1),label)
                val_loss+=loss.item()
            val_loss = val_loss/float(len(self.val_loader))
            val_loss_vs_epoch.append([val_loss])
            if val_loss<best_val_loss:                          # Best Model
                counter = 0
                best_val_loss = val_loss
                best_model = self.model
                torch.save(best_model.cpu().state_dict(),self.save_path+'/best_model.pth')
            # SAVE IMAGES
            mkdir(self.save_path+'/during_validation/images')
            mkdir(self.save_path+'/during_validation/labels')
            mkdir(self.save_path+'/during_validation/predicted')
            mkdir(self.save_path+'/during_validation/uncertainties')
            output_classes = torch.max(output,dim=1)[1]
            uncertainties = torch.max(F.softmax(output,dim=1),dim=1)[0]
            if epoch == 0:
                save_image(input[0],self.save_path+'/during_validation/images/%03d.png'%(epoch))
                save_image((label.float()/float(self.noc))[0],self.save_path+'/during_validation/labels/%03d.png'%(epoch))
            save_image((output_classes.float()/float(self.noc))[0],self.save_path+'/during_validation/predicted/%03d.png'%(epoch))
            save_image(uncertainties[0],self.save_path+'/during_validation/uncertainties/%03d.png'%(epoch))
            print ('EPOCH:',epoch,',TRAIN LOSS: ',train_loss,',VAL LOSS: ',val_loss)
            if epoch>100:
                counter+=1
            if (counter >=10):
                break
        with open (self.save_path+'/train_loss_vs_epoch.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(train_loss_vs_epoch)
        with open (self.save_path+'/val_loss_vs_epoch.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(val_loss_vs_epoch)
        print('TRAINING FINISHED')
        return best_model

class Tester():
    def __init__(self,trained_model,test_loader,save_path,noc):
        self.test_loader = test_loader
        self.save_path = save_path
        self.noc = noc
        self.model = trained_model
    def test(self):
        self.model = self.model.train(False)
        print('TESTING')
        image_id = 1
        probabilities=[]
        iou = np.array([0.0]*self.noc)
        acc = 0.0
        precision = np.array([0.0]*self.noc)
        recall = np.array([0.0]*self.noc)
        f1 = np.array([0.0]*self.noc)
        c_m = np.zeros((self.noc,self.noc))
        for input,label in self.test_loader:
            print (image_id)
            input,label = input.cuda(),label.cuda()
            output = self.model(input)
            output_classes = torch.max(output,dim=1)[1]
            uncertainties = torch.max(F.softmax(output,dim=1),dim=1)[0]
            probabilities.append(F.softmax(output,dim=1).cpu().data.numpy())
            mkdir(self.save_path+'/images')
            mkdir(self.save_path+'/labels')
            mkdir(self.save_path+'/predicted')
            mkdir(self.save_path+'/uncertainties')
            for indx in range(input.shape[0]):
                save_image(input[indx],self.save_path+'/images/%06d.png'%(image_id))
                save_image((label.float()/float(self.noc))[indx],self.save_path+'/labels/%06d.png'%(image_id))
                save_image((output_classes.float()/float(self.noc))[indx],self.save_path+'/predicted/%06d.png'%(image_id))
                save_image(uncertainties[indx],self.save_path+'/uncertainties/%06d.png'%(image_id))
                image_id+=1
            c_m = calc_conf(label,output_classes,self.noc,c_m)
        iou,precision,recall,f1,acc= calc_measures(self.noc,c_m)
        probabilities = torch.Tensor(np.concatenate(probabilities,axis=0))
        miou = iou.mean()
        mAP = precision.mean()
        mAR = recall.mean()
        mF1 = f1.mean()
        with open (self.save_path+'/iou_class.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(iou.reshape(iou.shape[0],1))
        with open (self.save_path+'/f1_class.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(f1.reshape(f1.shape[0],1))
        with open (self.save_path+'/precision_class.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(precision.reshape(precision.shape[0],1))
        with open (self.save_path+'/recall_class.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(recall.reshape(recall.shape[0],1))
        with open (self.save_path+'/confusion_matrix.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(c_m)
        results = [['miou',miou],['mAP',mAP],['mAR',mAR],['acc',acc],['mF1',mF1]]
        with open (self.save_path+'/summary.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(results)
        torch.save(probabilities,self.save_path+'/probabilities.pth')
        print('TESTING FINISHED')






