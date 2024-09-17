import os
import json

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from data.utils import pre_caption

class flip_pretrain(Dataset):
    def __init__(self, ann_root, image_root, transform): 
        ann_files = sorted(os.listdir(ann_root), 
                        key = lambda x: int(x.split("_")[-1].rstrip(".json")))

        self.annotation = []
        self.image_root = image_root

        for f in ann_files:
            print('loading ' + f)
            ann = json.load(open(os.path.join(ann_root, f), 'r'))
            self.annotation += ann
        self.transform = transform
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index] 
        image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')   
        image = self.transform(image)
        caption = pre_caption(ann['caption'][0], 50) 
        
        return image, caption


class flip_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=50, prompt=''):        
       
        filename = 'flip_align_split_00000.json'        
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
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
        
        caption = self.prompt + pre_caption(ann['caption'][0], self.max_words) 
        return image, caption, self.img_ids[ann['image_id']] 
    

class flip_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        filenames = {'val':'flip_align_split_00030.json', 'test':'flip_align_split_00031.json'}        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.annotation = self.annotation ##change it ************************************** [:5000]
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
       
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        print(ann['image'])
        print(img_id)
        return image, int(img_id)   
    
    
class flip_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  

        filenames = {'val':'flip_align_split_00030.json','test':'flip_align_split_00031.json'}
                
        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]),'r'))
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
            
            for i, caption in enumerate(ann['caption']):
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
    