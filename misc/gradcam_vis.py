import argparse
import os, random 
import ruamel.yaml as yaml
import numpy as np
from pathlib import Path

from PIL import Image
import cv2, json 
import numpy as np
from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import transforms

from models.blip_itm import blip_itm
import utils
from data import create_dataset, create_sampler, create_loader
from data.utils import pre_caption

font = {'font.family' : 'Times New Roman',
        'font.size'   : 15}
plt.rcParams.update(font)

def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()

    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')

    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()

    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)

    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap


def load_img_text_pair(max_word):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform = transforms.Compose([
        transforms.Resize((384,384),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])  

    image_path = 'datasets/celeba/images/61/981.jpg'
    image_pil = Image.open(image_path).convert('RGB')   
    image = transform(image_pil).unsqueeze(0)  

    caption = "This man is smiling and has black hair, bushy eyebrows, goatee, mustache, big lips, big nose, bags under eyes."
    caption = pre_caption(caption, max_word) 
    image = image.cuda()
    return image, image_path, caption 


def visualize_gradcam(model, text_input, image_path, gradcam, max_word):
    num_image = len(text_input.input_ids[0]) 
    fig, ax = plt.subplots(6, 5, figsize=(15,5*num_image))

    rgb_image = cv2.imread(image_path)[:, :, ::-1]
    rgb_image = np.float32(rgb_image) / 255

    #ax[0].imshow(rgb_image)
    #ax[0].set_yticks([])
    #ax[0].set_xticks([])
    #ax[0].set_xlabel("Image")
    k = 0
    l = 0
    for i,token_id in enumerate(text_input.input_ids[0][1 : max_word]):
        word = model.tokenizer.decode([token_id])
        gradcam_image = getAttMap(rgb_image, gradcam[i+1])
        ax[k, l].imshow(gradcam_image)
        ax[k, l].set_yticks([])
        ax[k, l].set_xticks([])
        ax[k, l].set_xlabel(word)
        if (i + 1) % 5 == 0:
            k += 1
            l = 0
        else: l += 1

    plt.show()


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            

    #### Model #### 
    print("Creating model")
    model = blip_itm(pretrained=config['pretrained'], 
                           image_size=config['image_size'], 
                           max_word = args.max_word,
                           img_encoder=config['img_encoder'])

    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    model.eval()
    block_num = 8
    model.text_encoder.encoder.layer[block_num].crossattention.self.save_attention = True

    #### Dataset #### 
    print("Creating image-text pair")
    #ann_root = os.path.join("datasets", args.dataset)
    #annotation = json.load(open(os.path.join(ann_root, "annotation", "val.json"),'r'))
    #for ann in annotation[220:400:5]:
    #    img_path = os.path.join(ann_root, "images", ann["image"]) 
    #    c = 0
    #    for cap in ann["caption"]:

    image, image_path, caption  = load_img_text_pair(args.max_word)

    output, text = model(image, caption, match_head='itm', return_text=True)
    loss = output[:,1].sum()
    model.zero_grad()
    loss.backward()    
    
    with torch.no_grad():
        mask = text.attention_mask.view(text.attention_mask.size(0),1,-1,1,1)

        grads=model.text_encoder.encoder.layer[block_num].crossattention.self.get_attn_gradients()
        cams=model.text_encoder.encoder.layer[block_num].crossattention.self.get_attention_map()

        cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 24, 24) * mask
        grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 24, 24) * mask

        gradcam = cams * grads
    gradcam = gradcam[0].mean(0).cpu().detach().numpy()
    visualize_gradcam(model, text, image_path, gradcam,  args.max_word)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--dataset', default='celeba')         
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    args = parser.parse_args()
    args.max_word = 31
    args.config = f'./configs/retrieval_{args.dataset}.yaml'
    args.output_dir = f'output/retrieval_{args.dataset}'

    
    yml = yaml.YAML(typ='rt')
    config = yml.load(open(args.config, 'r'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    yd = yaml.YAML(typ='unsafe', pure=True)
    yd.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    args.image_root = config['image_root']
    main(args, config)
    """
    python3 gradcam_vis.py --dataset celeba
    """