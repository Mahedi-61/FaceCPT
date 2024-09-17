import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.facecpt_pretrain import blip_pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50,       fmt='{value:.7f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))    
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50,  fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    data_loader.sampler.set_epoch(epoch)

    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if epoch==0:
            warmup_lr_schedule(optimizer, 
                               i, 
                               config['warmup_steps'], 
                               config['warmup_lr'], 
                               config['init_lr'])
            
        optimizer.zero_grad()
        image = image.to(device,non_blocking=True)
        
        # ramp up alpha in the first 2 epochs
        alpha = config['alpha'] * min(1, (epoch * len(data_loader) + i) / (2 * len(data_loader))) 

        loss_ita, loss_itm, loss_lm = model(image, caption, alpha = alpha)  
        loss = loss_ita + loss_itm + loss_lm  

        loss.backward()
        optimizer.step()    

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Model #### 
    print("Creating model")
    model = blip_pretrain(image_size=config['image_size'], 
                            vit=config['vit'], 
                            queue_size=config['queue_size'])

    model = model.to(device)   
    optimizer = torch.optim.AdamW(params=model.parameters(), 
                                lr=config['init_lr'], 
                                weight_decay=config['weight_decay'])
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    

        msg = model.load_state_dict(state_dict, strict=False)    
        print("missing keys in saved facecpt checkpont: ")
        print(msg)

        optimizer.load_state_dict(checkpoint['optimizer'])
        #start_epoch = checkpoint['epoch'] + 1                
        print('resume checkpoint from %s' % args.checkpoint)

    else:
        # Allah help me
        print("loading previous pre-train models to save computation")
        blip_base = "weights/model_base.pth"
        cp_blip = torch.load(blip_base, map_location='cpu', weights_only=True)  
        blip_dict = cp_blip['model']    

        flip_base = "weights/flip_base.pth"
        cp_flip = torch.load(flip_base, map_location='cpu') 
        flip_dict = cp_flip['model']   

        common_key_set = (set(flip_dict.keys())).intersection(set(blip_dict.keys()))
        for key in list(common_key_set):
            blip_dict[key] = flip_dict[key]

        #print(len(list(common_key_set)))
        msg = model.load_state_dict(blip_dict, strict=False)    
        print("missing keys in saved facecpt checkpont: ")
        #print(msg)


    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset('pretrain', config, min_scale = 0.60)] #0.2
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()   
       
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)         

    data_loader = create_loader(datasets,samplers,
                                batch_size=[config['batch_size']], 
                                num_workers=[4], 
                                is_trains=[True], 
                                collate_fns=[None])[0]      
  
    start_epoch = 0
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                        device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    
        
    print("Start training")
    start_time = time.time()   

    for epoch in range(start_epoch, config['max_epoch']):        
        step_lr_schedule(optimizer, epoch, 
                        config['init_lr'], 
                        config['min_lr'], 
                        config['lr_decay_rate'])
                
        train_stats = train(model, data_loader, optimizer, epoch, device, config) 

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 
                                    'cp_pretrain_flip_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()        
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='./configs/pretrain.yaml')
    parser.add_argument('--output_dir', default='output/pretrain')  
    parser.add_argument('--checkpoint', default='') #output/pretrain/cp_pretrain_flip_00.pth
    parser.add_argument('--evaluate',   action='store_true')    
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--seed',       default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url',   default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed',default=True, type=bool)
    args = parser.parse_args()

    yml = yaml.YAML(typ='rt')
    config = yml.load(open(args.config, 'r'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yd = yaml.YAML(typ='unsafe', pure=True)
    yd.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    main(args, config)


"""
Running code
python3 -m torch.distributed.run --nproc-per-node=2 pretrain.py
Put gradeint update on ArcFace model = True
"""