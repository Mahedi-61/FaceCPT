import os, sys, random
import os.path as osp
import argparse, itertools
import numpy as np
import pprint, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm 

from types import SimpleNamespace
from data.fr_attr_dataset import FrTrainDataset, FrTestDataset
from data.utils import *
from data.modules import calculate_acc
from models import arcface
from models.iresnet import get_arcface
from models import focal_loss 
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def freeze_arcface(net):
    print("Freezing first two layers")
    other_layers = [ 'visual_encoder.conv1.weight', 'visual_encoder.bn1.weight', 
                     'visual_encoder.bn1.bias', 'visual_encoder.prelu.weight'] 

    for name, param in net.named_parameters():
        param.requires_grad = True

    # freezing visual encoder weights
    for name, param in net.named_parameters():
        if "visual_encoder.layer1." in name:
            param.requires_grad = False

        elif "visual_encoder.layer2." in name:
            param.requires_grad = False

        elif name in other_layers:
            param.requires_grad = False


def unfreeze_arcface(net):
    print("unfreezing weights of arcface first layers")
    for name, param in net.named_parameters():
        param.requires_grad = True


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class Trainer:
    def __init__(self, args):
        self.args = args 
        self.args.device = my_device
        self.model_type = args.model_type

        self.get_data_loader()
        self.build_models() 
        self.get_optimizer() 
    
        print("Loading training and valid data ...")
        self.criterion_fr = focal_loss.FocalLoss(gamma=2)


    def get_data_loader(self):
        self.args.split="train"
        train_ds = FrTrainDataset(args = self.args)

        self.train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=self.args.batch_size, 
            drop_last=True,
            num_workers=self.args.num_workers, 
            shuffle=True)

        self.test_sets = ['lfw', 'agedb', 'calfw', 'cfp_fp',  'cpl_fw']
        self.dict_test_loader = {}

        for dataset in self.test_sets:
            loader = torch.utils.data.DataLoader(
                FrTestDataset(args, dataset),
                batch_size = args.batch_size, 
                drop_last = False,
                num_workers = 4, 
                shuffle = False)

            self.dict_test_loader.update({dataset: loader})



    def save_models(self):
        name = 'cp_fr_ms1m_arc50_epoch_%d' % self.args.current_epoch
        state_path = os.path.join(self.args.output_dir , name)
        state = {"base_model": self.base_model.state_dict(), 
                 "fr_model": self.metric_fc.state_dict()}
        
        torch.save(state, state_path)


    def build_models(self):
        self.base_model = get_arcface(self.model_type)
        self.metric_fc = ArcMarginProduct(512, 
                                        self.args.num_classes, 
                                        s=64.0, 
                                        m=0.5, 
                                        easy_margin=False).to(my_device)

        if args.checkpoint_path:
            print("loading saved checkpoint: ", args.checkpoint_path) 
            state_dict = torch.load(args.checkpoint_path,  weights_only=True)
            self.base_model.load_state_dict(state_dict['base_model'])
            self.metric_fc.load_state_dict(state_dict['fr_model'])

        else:
            print("loading pretrained FaceCPT model: ", args.pretrained)
            checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
            cp_dict = checkpoint['model']

            state_dict = {}
            for key in cp_dict.keys():
                if key.startswith("visual_encoder."):
                    state_dict[key.replace("visual_encoder.", "")] = cp_dict[key]
            
            assert list(self.base_model.state_dict().keys()) == list(state_dict.keys()); "Keys Doesn't Match!!"
            msg = self.base_model.load_state_dict(state_dict, strict=False)
            print("missing keys: ", msg)
    
        self.base_model.to(my_device)
        self.metric_fc.to(my_device)


    def get_optimizer(self):
        params_base = [{"params": self.base_model.parameters(), 
                         "lr" : 0.04, 
                         "weight_decay" : 0.0005}]
        
        params_metric = [{"params": self.metric_fc.parameters(), 
                         "lr" : 0.04, 
                         "weight_decay" : 0.0005}]
        
        self.optimizer_base = torch.optim.SGD(params_base, momentum=0.9)
        self.optimizer_metric = torch.optim.SGD(params_metric, momentum=0.9)

        self.lrs_opt_base = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer_base, 
                                        step_size = 45,
                                        gamma=0.9997)


        self.lrs_opt_metric = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer_metric, 
                                        step_size = 45,
                                        gamma=0.9997)


    def evaluate_fr(self,):
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
                global_feat1  = self.base_model(img1)
                global_feat2  = self.base_model(img2)

                global_feat1_h  = self.base_model(img1_h)
                global_feat2_h  = self.base_model(img2_h)

                gf1 = torch.cat((global_feat1, global_feat1_h), dim=1)
                gf2 = torch.cat((global_feat2, global_feat2_h), dim=1)

                pred = cosine_sim(gf1, gf2)
                preds += pred.data.cpu().tolist()
                labels += pair_label.data.cpu().tolist()

                # update loop information
                loop.update(1)
                loop.set_postfix()

        loop.close()
        acc, _ =  calculate_acc(preds, labels, low_thresh=-1.0, interval=0.005)
        return acc 


    def test_on_multiple_datasets(self, ):
        acc = []
        for dataset in self.test_sets:
            print("\n############## evaluating %s dataset ################" % dataset)
            self.test_dl = self.dict_test_loader[dataset]
            acc.append(self.evaluate_fr())

        return sum(acc)/len(acc)


    def train_epoch(self):
        self.base_model.train()
        self.metric_fc.train()

        epoch = self.args.current_epoch 
        total_length = len(self.train_dl) * self.args.batch_size
        total_fr_l = 0

        loop = tqdm(total = len(self.train_dl))
        for imgs, label in self.train_dl:   
            imgs = imgs.to(my_device)
            label = label.to(my_device)
        
            gl_feats = self.base_model(imgs) 
            output = self.metric_fc(gl_feats, label)

            # zero grad
            self.optimizer_metric.zero_grad()
            self.optimizer_base.zero_grad() 

            # loss
            loss_fr = self.criterion_fr(output, label)
            loss_fr.backward()

            total_fr_l += loss_fr.item()

            # updating weights
            if epoch > self.args.freeze: self.optimizer_base.step()
            self.optimizer_metric.step() 

            # updating scheduler
            self.lrs_opt_base.step()
            self.lrs_opt_metric.step()

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{self.args.epochs}]')
            loop.set_postfix()

        loop.close()
        print(' | epoch {:3d} |' .format(self.args.current_epoch))
        print(f" Learning Rate for base model: {self.lrs_opt_base.get_last_lr()[0]}")
        print("FR loss: {:3.5f} ".format(total_fr_l / total_length))

   
    def train(self):
        freeze_arcface(self.base_model)
        self.val_acc = 0

        for epoch in range(0, self.args.epochs):
            self.args.current_epoch = epoch

            self.train_epoch()
            if epoch > 3: 
                unfreeze_arcface(self.base_model)

            if epoch  > self.args.valid_interval:
                print("Let's evaluate FR performance")
                acc = self.test_on_multiple_datasets()

                if acc > self.val_acc:
                    print("\nLet's save the model")
                    self.val_acc = acc
                    self.save_models()  

        print("saving last model")
        self.save_models()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',          dest="train",    help='train the pretrained model',   action='store_true')
    parser.set_defaults(train = True)
    parser.add_argument('--evaluate',       dest="train",    help='evaluate the pretrained model',action='store_false')
    

    parser.add_argument('--dataset',        type=str,   default="ms1m",         help='ms1m | vgg')
    parser.add_argument('--batch_size',     type=int,   default=256,            help='batch size')
    parser.add_argument('--epochs',         type=int,   default=11,             help='Number of epochs') 

    parser.add_argument('--model_type',     type=str,   default="arcface_50",   help='architecture of the model: arcface | cosface')
    parser.add_argument('--checkpoint_path',type=str,   default="output/fr_ms1m/cp_fr_ms1m_arc50_epoch_7",    help='path of the saved cp')
    parser.add_argument('--freeze',         type=int,   default=1,              help='Number of epoch pretrained model frezees')

    parser.add_argument('--s',             type=float,   default=64.0,          help='arcface s')
    parser.add_argument('--m',             type=float,   default=0.5,           help='arcface margin') 
    parser.add_argument('--valid_interval', type=int,    default=5,              help='valid (epochs)')
    return  parser.parse_args(argv)


vgg_cfg = SimpleNamespace(
    num_classes = 8631
)

ms1m_cfg = SimpleNamespace(
    num_classes = 85742
)

setup_cfg = SimpleNamespace(
    pretrained = "output/pretrain/cp_pretrain_flip_01.pth", 

    metric = "arc_margin", 
    loss = "focal_loss", 
    use_se = False,
    manual_seed = 61,
    num_workers = 4,
    is_ident = False,
)


if __name__ == "__main__":
    c_args = parse_arguments(sys.argv[1:])

    if c_args.dataset == "ms1m":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **ms1m_cfg.__dict__)

    elif c_args.dataset == "vgg":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **vgg_cfg.__dict__)
    
    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.data_dir = os.path.join("./datasets", args.dataset)

    args.output_dir = f'output/fr_{args.dataset}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    t = Trainer(args)
    print("start training ...")

    if args.train == True:
        t.train()
    else:
        t.test_on_multiple_datasets()
    

"""
RUN the code
python3 train_fr.py  --dataset ms1m --evaluate
"""