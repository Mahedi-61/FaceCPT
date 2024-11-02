import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path
from PIL import Image 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader 
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip_itm import blip_itm
import utils
from data.utils import *



class caption_sim_dataset(Dataset):
    def __init__(self, transform, image_root, ann_root, max_word, split):  
        filenames = {'test':'test.json'}        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_word = max_word
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)     
        caption = pre_caption(ann['caption'][0], self.max_word) ##################### only first caption      
        return image, caption


def do_dis_plot(preds, labels, args):
    y_pos = []
    y_neg = []

    for i, label in enumerate(labels):
        if label == 1:
            y_pos.append(preds[i])
        elif label == 0:
            y_neg.append(preds[i])

    y_neg = y_neg [:len(y_pos)]

    with np.load("adaface_resnet_18.npz") as file:
        y_pos_18 = file["x"]
        y_neg_18 = file["y"]

    #np.savez("adaface_resnet_18.npz", x=y_pos, y=y_neg)
    
    plt.figure()
    df = pd.DataFrame({"y_pos" : y_pos, "y_pos_18" : y_pos_18, 
                       "y_neg" : y_neg, "y_neg_18" : y_neg_18})
    
    chart = sns.displot(data=df,  kind='kde', fill=True, height=5, aspect=1.5)
    plt.savefig("result.eps")
    

def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                     (0.26862954, 0.26130258, 0.27577711))
    
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),
                            interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  

    #### Dataset #### 
    print("Creating captioning dataset")
    test_dataset = caption_sim_dataset(transform_test, 
                                        config['image_root'], 
                                        config['ann_root'], 
                                        args.max_word,
                                        'test') 

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        sampler = torch.utils.data.DistributedSampler(test_dataset, 
                                                      num_replicas=num_tasks, 
                                                      rank=global_rank, 
                                                      shuffle=False)    
    else:
        sampler = None
    
    test_loader = DataLoader(
                    test_dataset,
                    batch_size= 64,
                    num_workers=4,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=False,
                    collate_fn=None,
                    drop_last=False)   

    #### Model #### 
    print("Creating model")
    print(config['pretrained'])
    model = blip_itm(pretrained=config['pretrained'], 
                           image_size=config['image_size'], 
                           max_word = args.max_word,
                           img_encoder=config['img_encoder'])

    model = model.to(device)   
   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    print("Start Testing")
    start_time = time.time()    

    model.eval() 
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Similarity Matching'
    print_freq = 10
    result = []

    with torch.no_grad():
        for image, caption in metric_logger.log_every(test_loader, print_freq, header): 
            image = image.to(device) 
            cos_metric = model(image, caption, match_head='itc') 
            result += cos_metric.cpu().tolist()

    df = pd.DataFrame({"celeba_test": result})
    sns.kdeplot(df)
    plt.show()

    #dist.barrier()     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celeba')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    args = parser.parse_args()
    args.max_word = 40
    args.config = f'./configs/caption_{args.dataset}.yaml'
    
    args.output_dir = f'output/caption_{args.dataset}'

    yml = yaml.YAML(typ='rt')
    config = yml.load(open(args.config, 'r'))

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    yd = yaml.YAML(typ='unsafe', pure=True)
    yd.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))     
    config["pretrained"] = f"checkpoint/checkpoint_retrieval_best_celeba.pth"
    main(args, config)

    """
    python3 caption_similarity_matching.py --dataset celeba
    """