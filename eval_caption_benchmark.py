import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from models.facecpt import facecpt_decoder
import utils
from data import create_dataset, create_sampler, create_loader
from data.utils import *


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    model.eval() 

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'caption generation:'
    print_freq = 20

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        image = image.to(device)       
        captions = model.generate(image, 
                                  sample=True, 
                                  num_beams=config['num_beams'], 
                                  max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            id = img_id.split("/")[0]
            result.append({"image": img_id, "caption": [caption], "image_id": id})
    return result
  

def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    test_dataset = create_dataset(f'caption_benchmark', config)  


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([test_dataset], [False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    test_loader = create_loader([test_dataset],
                    samplers,
                    batch_size=[config['batch_size']] * 4,
                    num_workers=[4],
                    is_trains=[False], 
                    collate_fns=[None,None,None]) [0] # (indexing to get from list)        
    
    #### Model #### 
    print("Creating model")
    model = facecpt_decoder(pretrained=config['pretrained'],
                            image_size=config['image_size'], 
                            vit=config['vit'], 
                            prompt=config['prompt'])

    model = model.to(device)   
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                            device_ids=[args.gpu], 
                                            find_unused_parameters=True)
        model_without_ddp = model.module    
    
    print("Start Generating Captions")
    start_time = time.time() 

    test_result = evaluate(model_without_ddp, test_loader, device, config)  
    save_result(test_result, args.result_dir, 
                    'test_caption_%s'%args.dataset, 
                    remove_duplicate='image')  

    #dist.barrier()     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str)) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='lfw')  #calfw
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    args = parser.parse_args()
    args.config = f'./configs/caption_benchmark.yaml'
    args.output_dir = f'output/caption_benchmark'

    args.result_dir = os.path.join(args.output_dir, args.dataset, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yml = yaml.YAML(typ='rt')
    config = yml.load(open(args.config, 'r'))
    config["image_root"] = os.path.join("../FaceCPT/datasets", args.dataset, "images")
    config["ann_root"] = os.path.join("../FaceCPT/datasets", args.dataset, "annotation") 


    yd = yaml.YAML(typ='unsafe', pure=True)
    yd.dump(config, open(os.path.join(args.output_dir, args.dataset, 'config.yaml'), 'w'))     

    main(args, config)

    """
    python3 -m torch.distributed.run --nproc-per-node=2 eval_caption_benchmark.py --dataset lfw
    """