import os
import json

from torch.utils.data import Dataset
from PIL import Image
from data.utils import pre_caption

class dataset_caption_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=50, prompt=''):        
       
        filename = 'train.json'
        print("loading josn file: ", os.path.join(ann_root, filename))        
        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt + pre_caption(ann['caption'], self.max_words) 
        return image, caption, self.img_ids[ann['image_id']] 


class dataset_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        filenames = {'val':'val.json', 'test':'test.json'} 
        print("loading josn file: ", os.path.join(ann_root, filenames[split]))       
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        return image, ann['image'] 