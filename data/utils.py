import re
import json
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.distributed as dist
import utils
import evaluate
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A 


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.belu_1 = 0
        self.belu_2 = 0
        self.belu_3 = 0
        self.belu_4 = 0

        self.rougeL = 0
        self.meteor = 0
        self.count = 0

    def update(self, belu_1, belu_2, belu_3, belu_4, rougeL, meteor):
        self.belu_1 += belu_1
        self.belu_2 += belu_2
        self.belu_3 += belu_3
        self.belu_4 += belu_4
    
        self.rougeL += rougeL
        self.meteor  += meteor 
        self.count += 1

    def average(self):
        self.belu_1  = self.belu_1 / self.count
        self.belu_2  = self.belu_2 / self.count
        self.belu_3  = self.belu_3 / self.count
        self.belu_4  = self.belu_4 / self.count

        self.rougeL = self.rougeL / self.count
        self.meteor  = self.meteor / self.count


def cap_metrics(gen_caption, ref_caption):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')
    meter = AverageMeter()

    keys = list(gen_caption.keys())

    gcap = [gen_caption[key] for key in keys]
    rcap = [ref_caption[key] for key in keys]
    b1 = bleu.compute(predictions = gcap, references= rcap, max_order=1)
    b2 = bleu.compute(predictions = gcap, references= rcap, max_order=2)
    b3 = bleu.compute(predictions = gcap, references= rcap, max_order=3)
    b4 = bleu.compute(predictions = gcap, references= rcap, max_order=4)

    rL = rouge.compute(predictions = gcap, references= rcap)
    m  = meteor.compute(predictions = gcap, references= rcap)

    meter.update(b1["bleu"], b2["bleu"],   b3["bleu"], 
                 b4["bleu"], rL["rougeL"], m["meteor"])

    meter.average()
    return {"BLEU@4" : meter.belu_4, 
            "rougeL" : meter.rougeL, 
            "METEOR" : meter.meteor}


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption



def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))
    dist.barrier()

    # combine results from all processes
    if utils.is_main_process():   
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)
    return final_result_file



def caption_eval(ann_root, results_file, split):
    filenames = {'val':'val.json', 'test':'test.json'}         
    annotation_file = os.path.join(ann_root, filenames[split])

    ann_file = json.load(open(annotation_file,'r'))
    res_file = json.load(open(results_file,'r'))

    ref_caption = {file['image']: file['caption'] for file in ann_file}
    gen_caption = {file['image_id']: file['caption'] for file in res_file}

    s1 = set(list(gen_caption.keys()))
    s2 = set(list(ref_caption.keys()))
    assert s1 == s2, "Mismatch !!"

    eval_dict = cap_metrics(gen_caption, ref_caption)

    # calculating CIDEr score
    c = Cider()
    gen_caption = {file['image_id']: [file['caption']] for file in res_file}
    cscore, _ = c.compute_score(ref_caption, gen_caption)

    # calculating SPICE score
    s = Spice()
    sscore, _ = s.compute_score(ref_caption, gen_caption)
    print("BLEU@4: ", eval_dict['BLEU@4'])
    print("rougeL: ", eval_dict['rougeL'])
    print("METEOR: ", eval_dict['METEOR'])

    print("SPICE: ", sscore)
    print("CIDEr: ", cscore)


    eval_dict.update({"CIDEr" :cscore, "SPICE" : sscore})
    return eval_dict



def transform_images(img_path, split):
    img = np.array(Image.open(img_path).convert('RGB')) 
    sample_transforms = [
        A.HorizontalFlip(),
        A.ColorJitter(),
        A.Rotate(15),
        A.RandomBrightnessContrast(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=30, p=0.5),
        A.HueSaturationValue(p = 0.3)
    ]

    train_transforms = A.Compose([
        *sample_transforms,
        A.Normalize(mean = [0.5, 0.5, 0.5],  
                    std =  [0.5, 0.5, 0.5],  
                    always_apply=True),
        ToTensorV2()
    ])

    valid_transforms = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], 
                    std =[0.5, 0.5, 0.5], always_apply=True),
        ToTensorV2()
    ])

    if split == "train": tfms = train_transforms
    elif split == "test" or split == "valid":  tfms = valid_transforms

    img = tfms(image=img)["image"] 
    return img


def do_flip_test_images(img_path):
    img = np.array(Image.open(img_path).convert('RGB')) 
    tfms = A.Compose([
        A.HorizontalFlip(p = 1),
        A.Normalize(mean=[0.5, 0.5, 0.5], 
                    std =[0.5, 0.5, 0.5], always_apply=True),
        ToTensorV2()
    ])

    img = tfms(image=img)["image"] 
    return img


all_attributes = ["5_o_Clock_Shadow",	"Arched_Eyebrows",	"Attractive",	"Bags_Under_Eyes",	"Bald",	
                "Bangs",	"Big_Lips",	"Big_Nose",	"Black_Hair", "Blond_Hair",	
                "Blurry",	"Brown_Hair",	"Bushy_Eyebrows",	"Chubby",	"Double_Chin",
                "Eyeglasses",	"Goatee",	"Gray_Hair",	"Heavy_Makeup",	"High_Cheekbones",
                "Male",	"Mouth_Slightly_Open",	"Mustache", "Narrow_Eyes", "No_Beard",
                "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",	
                "Sideburns", "Smiling",	"Straight_Hair", 	"Wavy_Hair",	"Wearing_Earrings",
                "Wearing_Hat",	"Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]