import os, sys, random
import os.path as osp
import argparse
import numpy as np

import torch
import torch.nn as nn
import pprint
from tqdm import tqdm 
from datetime import datetime 

#ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
#sys.path.insert(0, ROOT_PATH)

from types import SimpleNamespace
from data.utils import *
from data.modules import calculate_acc, calculate_scores
from models import arcface 
from data import FrTestDataset


class Evaluate:
    def __init__(self, args):
        self.test_ds = FrTestDataset(args)

        self.test_dl = self.get_data_loader()
        args.device = my_device
        self.build_models()

    def build_models(self, ):
        if args.architecture == "ir_50":
            self.base_model = arcface.iresnet50(pretrained=False, progress=True)
            pretrained_weight_path = "checkpoint/fr/cp_fr_celeba_dialog.pth"

        elif args.architecture == "ir_101":
            self.base_model = arcface.iresnet101(pretrained=False, progress=True)
            pretrained_weight_path = args.weights_arcface_101
    
        if args.checkpoint_path:
            print("loading from saved checkpoint: ", args.checkpoint_path) 
            state_dict = torch.load(args.checkpoint_path,  weights_only=True)
            self.base_model.load_state_dict(state_dict['base_model'])

        else:
            print("loading pretrained FaceCPT model: ", pretrained_weight_path)
            checkpoint = torch.load(pretrained_weight_path, map_location="cpu", weights_only=False)

            cp_dict = checkpoint['model']
            state_dict = {}

            for key in cp_dict.keys():
                if key.startswith("visual_encoder."):
                    state_dict[key.replace("visual_encoder.", "")] = cp_dict[key]

            msg = self.base_model.load_state_dict(state_dict, strict=False)
            print("missing keys: ", msg)
    
        self.base_model.to(my_device)


    def get_data_loader(self,):
        return torch.utils.data.DataLoader(
                self.test_ds, 
                batch_size = args.batch_size, 
                drop_last = False,
                num_workers = 4, 
                shuffle = False)
   

    def test(self, args):
        self.base_model.eval()
        preds = []
        labels = []

        loop = tqdm(total = len(self.test_dl))
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        with torch.no_grad():
            for step, data in enumerate(self.test_dl, 0):
                img1, img2, img1_h, img2_h, pair_label = data 
                
                img1 = img1.to(my_device)
                img2 = img2.to(my_device)

                img1_h = img1_h.to(my_device)
                img2_h = img2_h.to(my_device)
                pair_label = pair_label.to(my_device)
            
                # get global and local image features from COTS model
                if args.model_type == "arcface":
                    global_feat1,  _ = self.base_model(img1)
                    global_feat2,  _ = self.base_model(img2)

                    global_feat1_h,  _ = self.base_model(img1_h)
                    global_feat2_h,  _ = self.base_model(img2_h)

                gf1 = torch.cat((global_feat1, global_feat1_h), dim=1)
                gf2 = torch.cat((global_feat2, global_feat2_h), dim=1)

                pred = cosine_sim(gf1, gf2)
                preds += pred.data.cpu().tolist()
                labels += pair_label.data.cpu().tolist()

                # update loop information
                loop.update(1)
                loop.set_postfix()

        loop.close()
        calculate_acc(preds, labels, low_thresh=-1.0, interval=0.005)



setup_cfg = SimpleNamespace(
    weights_arcface_18 = "./weights/arcface_ir18_ms1mv3.pth", 
    weights_arcface_50 = "./weights/arcface_ir50_ms1mv3.pth", 
    weights_arcface_101= "./weights/arcface_ir101_ms1mv3.pth", 

    metric = "arc_margin", 
    easy_margin = False,
    loss = "focal_loss", 
    use_se = False,
    manual_seed= 61
)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=str,   default="lfw",               help='Name of the datasets lfw | calfw | agedbd')
    parser.add_argument('--batch_size',    type=int,   default=4,                   help='Batch size')
    parser.add_argument('--architecture',  type=str,   default="ir_50",             help='iResNet Architecture 18|50|101')
    parser.add_argument('--model_type',    type=str,   default="arcface",           help='architecture of the model: arcface | adaface | magface')
    parser.add_argument('--test_file',     type=str,   default="test_pairs.txt",    help='Name of the test list file')
    parser.add_argument('--checkpoint_path',type=str,  default='', help="checkpoint directory")

    return  parser.parse_args(argv)


if __name__ == "__main__":
    c_args = parse_arguments(sys.argv[1:])
    args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__)

    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.data_dir = os.path.join("./datasets", args.dataset)
    args.test_ver_list = os.path.join(args.data_dir, args.test_file)
    #pprint.pp(args)
    
    print("dataet directory: ", args.data_dir)
    eval = Evaluate(args)
    eval.test(args)


"""
RUN THE CODE
python3 eval_fr_sota_benchmark.py  --architecture ir_50 --dataset lfw
"""