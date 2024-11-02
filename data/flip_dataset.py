import os
import json
import torch 
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from data.utils import pre_caption

class flip_pretrain(Dataset):
    def __init__(self, ann_root, image_root, max_words, transform): 
        ann_files = sorted(os.listdir(ann_root), 
                        key = lambda x: int(x.split("_")[-1].rstrip(".json")))

        self.annotation = []
        self.image_root = image_root
        self.max_words = max_words
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
        caption = pre_caption(ann['caption'][0],  self.max_words) 
        attr_vec = torch.as_tensor(ann['attr_vec'], dtype=torch.float32)
        return image, caption, attr_vec
