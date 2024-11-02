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
from data.fr_attr_dataset import AttrDataset, FrTestDataset
from data.utils import *
from data.modules import calculate_acc
from models import arcface
from models.iresnet import get_image_encoder
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, args):
        self.args = args 
        self.args.device = my_device
        self.model_type = args.model_type

        self.get_data_loader()
        self.build_models() 
        self.get_optimizer() 
    
        print("Loading training and valid data ...")
        self.criterion_attr = nn.BCEWithLogitsLoss()


    def get_data_loader(self):
        train_ds = AttrDataset(split="train", args = self.args)
        print(f"###### Dataset: {args.dataset} #########")

        self.train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=self.args.batch_size, 
            drop_last=True,
            num_workers=self.args.num_workers, 
            shuffle=True)

        valid_ds =  AttrDataset(split="valid", args = self.args)
        self.valid_dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=self.args.batch_size, 
            drop_last=False,
            num_workers=self.args.num_workers, 
            shuffle=False)

        test_ds =  AttrDataset(split="test", args = self.args)
        self.test_dl = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=self.args.batch_size, 
            drop_last=False,
            num_workers=self.args.num_workers, 
            shuffle=False)


    def save_models(self):
        name = 'cp_attribute_celeba_dialog_arc50_%d.pth' % self.args.current_epoch
        state_path = os.path.join(args.output_dir, name)
        state = {"base_model" : self.base_model.state_dict(), 
                 "attr_model" : self.attr_model.state_dict()
                }
        
        torch.save(state, state_path)


    def build_models(self):
        self.attr_model = arcface.AttributeModel(num_attributes=40, args=args)
        self.base_model, img_width = get_image_encoder(self.model_type)
    
        if args.checkpoint_path:
            print("loading saved checkpoint: ", args.checkpoint_path) 
            state_dict = torch.load(args.checkpoint_path,  weights_only=True)
            self.base_model.load_state_dict(state_dict['base_model'])
            self.attr_model.load_state_dict(state_dict['attr_model'])

        else:
            print("loading pretrained FaceCPT model: ", args.pretrained)
            checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
            cp_dict = checkpoint['model']

            state_dict = {}
            for key in cp_dict.keys():
                if key.startswith("visual_encoder."):
                    state_dict[key.replace("visual_encoder.", "")] = cp_dict[key]
            
            assert list(self.base_model.state_dict().keys()) == list(state_dict.keys()); "Keys Doesn't Match!!"
            #msg = self.base_model.load_state_dict(state_dict, strict=False)

            msg = self.base_model.load_state_dict(state_dict, strict=False)
            print("missing keys: ", msg)
    
        self.base_model.to(my_device)
        self.attr_model.to(my_device)


    def get_optimizer(self):
        params_model = [{"params": self.base_model.parameters(), 
                         "lr" : 0.05, 
                         "weight_decay" : 0.0005}]
                
        params_attr = [{"params": self.attr_model.parameters(), 
                        "lr" : 0.002, 
                        "weight_decay" : 0.00005}]
        
        self.optimizer_model = torch.optim.SGD(params_model, momentum=0.9)
        self.optimizer_attr = torch.optim.Adam(params_attr)

        self.lrs_optimizer_model = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer_model, 
                                        step_size = 5,
                                        gamma=0.99) #0.9990 (dialog), 0.99(lfw_a)
        
        self.lrs_optimizer_attr = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer_attr, 
                                        step_size = 5,
                                        gamma=0.995) #09997 (dialog), 0.995(lfw_a)


    def evaluate_attr(self, eval_dl):
        self.base_model.eval()
        self.attr_model.eval()

        preds = []
        labels = []

        loop = tqdm(total = len(eval_dl))
        for imgs, attr_vec, cls_id in eval_dl:
            imgs = imgs.to(my_device)
            attr_vec = attr_vec.to(my_device)

            # get global and local image features
            gl_feats = self.base_model(imgs) 
            pred = torch.nn.functional.sigmoid(self.attr_model(gl_feats))

            bs, num_atrr = pred.size()
            pred = pred.reshape(bs*num_atrr)
            attr_vec = attr_vec.reshape(bs*num_atrr) 

            preds += pred.data.cpu().tolist()
            labels += attr_vec.data.cpu().tolist()

            loop.update(1)
            loop.set_postfix()

        loop.close()
        print("\n calculating attribution prediction accuracy: ")
        acc, _ = calculate_acc(preds, labels, low_thresh=0.0, interval=0.05) 
        return acc 


    def train_epoch(self):
        self.base_model.train()
        self.attr_model.train()

        epoch = self.args.current_epoch 
        total_length = len(self.train_dl) * self.args.batch_size
        total_attr_l = 0

        loop = tqdm(total = len(self.train_dl))

        for imgs, attr_vec, label in self.train_dl:
            imgs = imgs.to(my_device)
            attr_vec = attr_vec.to(my_device)
        
            gl_feats  = self.base_model(imgs) 
            pred_attrs = self.attr_model(gl_feats)

            # zero grad
            self.optimizer_attr.zero_grad()
            if epoch > self.args.freeze: self.optimizer_model.zero_grad() 

            # loss
            loss_attr =  self.criterion_attr(pred_attrs, attr_vec)
            loss_attr.backward()
            total_attr_l += loss_attr.item()

            # updating weights
            if epoch > self.args.freeze: 
                self.optimizer_model.step()
            self.optimizer_attr.step()

            # updating scheduler
            self.lrs_optimizer_model.step()
            self.lrs_optimizer_attr.step()

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{self.args.epochs}]')
            loop.set_postfix()

        loop.close()
        print(' | epoch {:3d} |' .format(self.args.current_epoch))
        print("model lr: ", self.lrs_optimizer_model.get_last_lr()[0])
        print("head lr: ", self.lrs_optimizer_attr.get_last_lr()[0])
        print("Attribute loss: {:3.5f} ".format(total_attr_l / total_length))

   
    def train(self):
        self.val_acc = 0
        for epoch in range(0, self.args.epochs):
            self.args.current_epoch = epoch

            self.train_epoch()
            
            if epoch  > self.args.valid_interval:
                print("\nLet's validate the model")
                acc = self.evaluate_attr(eval_dl = self.test_dl) ####################change

                if acc > self.val_acc:
                    print("\nLet's save the model")
                    self.val_acc = acc
                    self.save_models()  

        print("saving last model")
        self.save_models()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',          dest="train",    help='train the pretrained model',   action='store_true')
    parser.add_argument('--evaluate',       dest="train",    help='evaluate the pretrained model',action='store_false')
    parser.set_defaults(train = True)

    parser.add_argument('--dataset',      type=str,   default="lfw_a",    help='celeba_dialog|lfw_a')
    parser.add_argument('--batch_size',   type=int,   default=128,           help='batch size')
    parser.add_argument('--epochs',       type=int,   default=20,            help='Number of epochs') #15 (dialog), 20 (lfw_a)
    parser.add_argument('--model_type',   type=str,   default="arcface_50",  help='arch.: ArcFace (ResNet50, RestNet101)')
    return  parser.parse_args(argv)


setup_cfg = SimpleNamespace(
    pretrained = "output/pretrain/cp_pretrain_flip_00.pth", 
    checkpoint_path = "",
    valid_interval = 8,
    freeze = 5,  # 3 (celeba_dialog), 5 (lfw_a)
    use_se = False,
    manual_seed = 61,
    num_workers = 4
)

if __name__ == "__main__":
    c_args = parse_arguments(sys.argv[1:])

    if c_args.dataset == "celeba_dialog":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__) 

    elif c_args.dataset == "lfw_a":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__)
    
    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.data_dir = os.path.join("./datasets", args.dataset)
    args.ann_root = os.path.join(args.data_dir, "annotation")
    args.output_dir = f'output/attr_{args.dataset}'

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    t = Trainer(args)

    print("start training ...")
    if args.train == True:
        t.train()
    elif args.train == False:
        t.evaluate_attr(eval_dl = t.test_dl)


"""
RUN the code
python3 train_attribute.py --dataset celeba_dialog --evaluate
"""