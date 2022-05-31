import random
import os.path as path
import os
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import cv2
import numpy as np

from PIL import Image
from PIL.ImageOps import invert


class PollenDataset(BaseDataset):
    def name(self):
        return 'PollenDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print(opt)
       
        self.trainingImgsPaths = []
        self.referenceImgsPaths = []
      
        trainingPath =  path.join(self.root, "training")
        referencePath = path.join(self.root, "reference")

        for _, _, files in os.walk(trainingPath):
            for f in files:
                if (f.endswith(".bmp")):
                    self.trainingImgsPaths.append(path.join(trainingPath, f))

        for _, _, files in os.walk(referencePath):
            for f in files:
                if (f.endswith(".bmp")):
                    self.referenceImgsPaths.append(path.join(referencePath, f))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        self.shuffle_indices()

        #print the path and size
        print("training image  \t\tsize", len(self.trainingImgsPaths))
        print("reference \t\tsize", len(self.referenceImgsPaths))
        self.current_set_len = len(self.trainingImgsPaths)   


    def shuffle_indices(self):
        self.training_indices = list(range(len(self.trainingImgsPaths)))
        self.reference_indices = list(range(len(self.referenceImgsPaths)))

        if not self.opt.serial_batches:
            random.shuffle(self.training_indices)
            random.shuffle(self.reference_indices)

    def __getitem__(self, index):

        if index == 0:
            self.shuffle_indices()

        trainingImgName = self.trainingImgsPaths[self.training_indices[index % len(self.trainingImgsPaths)]]
        referenceImgName = self.referenceImgsPaths[self.reference_indices[index % len(self.referenceImgsPaths)]]
    
        A_img = cv2.imread(trainingImgName)
        # A_img = A_img.resize((32, 32))
        A_img = self.transform(A_img)           

        B_img = cv2.imread(referenceImgName)   
        B_img = self.transform(B_img)

        item = {}
        item.update({'A': A_img,
                     'A_paths': trainingImgName,
                 })
        
        item.update({'B': B_img,
                     'B_paths': referenceImgName,
                 })
        return item
        
    def __len__(self):
        return self.current_set_len
        
