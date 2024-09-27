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
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.face_retrieval import facecpt_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader


def test(model_without_ddp, test_loader, device, args, config):
    score_test_t2i = evaluation(model_without_ddp, test_loader, device, config) 

    if utils.is_main_process():
        test_result = itm_eval(score_test_t2i, 
                               test_loader.dataset.txt2img, 
                               test_loader.dataset.img2txt) 
        print(test_result)
        log_stats = {**{f'test_{k}': v for k, v in test_result.items()}}
        with open(os.path.join(args.output_dir, "test_evaluate.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")     



@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 

    metric_logger = utils.MetricLogger(delimiter="  ")
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []

    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', 
                                     truncation=True, 
                                     max_length=40, 
                                     return_tensors="pt").to(device) 
        
        text_output = model.text_encoder.bert(text_input.input_ids, 
                                         attention_mask = text_input.attention_mask, 
                                         mode='text')  
        
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    #text_ids[:,0] = model.tokenizer.enc_token_id
    
    image_feats = []
    image_embeds = []

    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image).unsqueeze(dim=1)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()

    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 

    
    # caption --> face 
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)), -100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)  
    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, "Evaluation")):
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        image_feats = image_feats.to(device)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)

        output = model.text_encoder.bert(text_ids[start+i].repeat(config['k_test'],1), 
                                attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                encoder_hidden_states = encoder_output,
                                encoder_attention_mask = encoder_att,                             
                                return_dict = True,
                                mode = "fusion",
                                )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 
    return score_matrix_t2i.cpu().numpy() 


@torch.no_grad()
def itm_eval(scores_t2i, txt2img, img2txt): 

    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        


    ir_mean = (ir1 + ir5 + ir10) / 3
    eval_result =  {'txt2img_r1': ir1,
                    'txt2img_r5': ir5,
                    'txt2img_r10': ir10,
                    'txt2img_r_mean': ir_mean}
    return eval_result


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
    print("Creating retrieval dataset")
    test_dataset = create_dataset('retrieval_benchmark', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([test_dataset], [False], num_tasks, global_rank) + [None, None]

    else:
        samplers = [None]

    test_loader = create_loader(
                    [test_dataset], 
                    samplers,
                    batch_size=[config['batch_size_train']] + [config['batch_size_test']] * 2,
                    num_workers=[2],
                    is_trains=[False], 
                    collate_fns=[None])[0]   

    #### Model #### 
    print("Creating model")
    model = facecpt_retrieval(pretrained=config['pretrained'], 
                            image_size=config['image_size'], 
                            vit=config['vit'], 
                            queue_size=config['queue_size'], 
                            negative_all_rank=config['negative_all_rank'])

    print("Start Testing")
    start_time = time.time()

    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                            device_ids=[args.gpu],
                                            find_unused_parameters=True)
        model_without_ddp = model.module   
        test(model_without_ddp, test_loader, device, args, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str)) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--dataset',    default='celeba')         
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--seed',       default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url',   default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed',default=True, type=bool)

    args = parser.parse_args()
    args.config = f'./configs/retrieval_benchmark.yaml'
    args.output_dir = os.path.join("output", "retrieval_benchmark",  args.dataset)

    yml = yaml.YAML(typ='rt')
    config = yml.load(open(args.config, 'r'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config["image_root"] = os.path.join("../FaceCPT/datasets", args.dataset, "images")
    config["ann_root"] = os.path.join("../FaceCPT/datasets", args.dataset, "annotation") 
    
    yd = yaml.YAML(typ='unsafe', pure=True)
    yd.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
    """
    python3 -m torch.distributed.run --nproc-per-node=2 eval_retrieval_benchmark.py --dataset celeba
    """