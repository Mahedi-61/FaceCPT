import os
import json

from torch.utils.data import Dataset
from PIL import Image
from data.utils import pre_caption

def get_max_words(dataset):
    if dataset == 'retrieval_celeba':
        return 40

    elif dataset == 'retrieval_celeba_dialog':
        return 40
        
    elif dataset == 'retrieval_face2text':
        return 60


class dataset_retrieval_train(Dataset):
    def __init__(self, transform, image_root, ann_root, dataset):        
       
        filename = 'train.json'        
        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        print("loading josn file: ", os.path.join(ann_root, filename))  

        self.transform = transform
        self.image_root = image_root
        self.max_words = get_max_words(dataset)      
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
        
        caption = pre_caption(ann['caption'], self.max_words) 
        return image, caption, self.img_ids[ann['image_id']] 
    
    
class dataset_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, dataset, split):  

        filenames = {'val':'val_retrieval.json', 'test':'test_retrieval.json'} 
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]),'r'))
        print("loading josn file: ", os.path.join(ann_root, filenames[split]))  

        max_words = get_max_words(dataset)
        self.transform = transform
        self.image_root = image_root
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            
            for i, caption in enumerate(ann['caption'][:2]): 
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index