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
#ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
#sys.path.insert(0, ROOT_PATH)

from types import SimpleNamespace
from data.fr_attr_dataset import FrAttrDataset, FrTestDataset
from data.utils import *
from data.modules import calculate_acc
from models import arcface
from models import focal_loss 
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.criterion_attr = nn.BCEWithLogitsLoss()
        self.criterion_fr = focal_loss.FocalLoss(gamma=2)

        #### for FR validation
        data_dir = os.path.join("./datasets", "lfw")
        args.test_ver_list  = os.path.join(data_dir, "test_pairs.txt")
        self.eval_fr_dl = torch.utils.data.DataLoader(
                FrTestDataset(args, data_dir),
                batch_size = args.batch_size, 
                drop_last = False,
                num_workers = 4, 
                shuffle = False)


    def get_data_loader(self):
        train_ds = FrAttrDataset(split="train", args = self.args)
        print(f"################ Dataset: {args.dataset}")

        self.train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=self.args.batch_size, 
            drop_last=True,
            num_workers=self.args.num_workers, 
            shuffle=True)
        

        valid_ds =  FrAttrDataset(split="valid", args = self.args)
        self.valid_dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=self.args.batch_size, 
            drop_last=False,
            num_workers=self.args.num_workers, 
            shuffle=False)

        test_ds =  FrAttrDataset(split="test", args = self.args)
        self.test_dl = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=self.args.batch_size, 
            drop_last=False,
            num_workers=self.args.num_workers, 
            shuffle=False)


    def save_models(self):
        save_dir = os.path.join("./weights", self.args.dataset)        
        os.makedirs(save_dir, exist_ok=True)

        name = 'fr_attr_%s_%d.pth' % (self.model_type, self.args.current_epoch)
        state_path = os.path.join(save_dir, name)
        state = {"base_model": self.base_model.state_dict(), 
                 "attr_model" : self.attr_model.state_dict(),
                 "fr_model": self.metric_fc.state_dict()}
        
        torch.save(state, state_path)


    def build_models(self):
        self.attr_model = arcface.AttributeModel(num_attributes=40, args=args)
        self.metric_fc = ArcMarginProduct(512, 
                                        self.args.num_classes, 
                                        s=64.0, 
                                        m=0.5, 
                                        easy_margin=False).to(my_device)

        if args.architecture == "ir_50":
            self.base_model = arcface.iresnet50(pretrained=False, progress=True)
            pretrained_weight_path = args.weights_arcface_50


        elif args.architecture == "ir_101":
            self.base_model = arcface.iresnet101(pretrained=False, progress=True)
            pretrained_weight_path = args.weights_arcface_101
    
        if args.checkpoint_path:
            print("loading saved checkpoint: ", args.checkpoint_path) 
            state_dict = torch.load(args.checkpoint_path,  weights_only=True)
            self.base_model.load_state_dict(state_dict['base_model'])
            self.attr_model.load_state_dict(state_dict['attr_model'])
            self.metric_fc.load_state_dict(state_dict['fr_model'])

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
        self.attr_model.to(my_device)
        self.metric_fc.to(my_device)


    def evaluate_fr(self,):
        self.base_model.eval()
        preds = []
        labels = []

        loop = tqdm(total = len(self.eval_fr_dl))
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        with torch.no_grad():
            for step, data in enumerate(self.eval_fr_dl, 0):
                img1, img2, img1_h, img2_h, pair_label = data 
                
                img1 = img1.to(my_device)
                img2 = img2.to(my_device)

                img1_h = img1_h.to(my_device)
                img2_h = img2_h.to(my_device)
                pair_label = pair_label.to(my_device)
            
                # get global and local image features from COTS model
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


    def get_optimizer(self):
        params_model = [{"params": self.base_model.parameters(), 
                         "lr" : 0.002, 
                         "weight_decay" : 0.0005}]
                
        params_attr = [{"params": self.attr_model.parameters(), 
                        "lr" : 0.001, 
                        "weight_decay" : 0.00005}]
        
        params_fr = [{"params": self.metric_fc.parameters(), 
                      "lr" : 0.01, 
                      "weight_decay" : 0.0005}]
        
        self.optimizer_model = torch.optim.SGD(params_model, momentum=0.9)
        self.optimizer_attr = torch.optim.Adam(params_attr)
        self.optimizer_fr = torch.optim.SGD(params_fr, momentum=0.9)

        self.lrs_optimizer_model = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer_model, 
                                        step_size = 5,
                                        gamma=0.999)
        
        self.lrs_optimizer_attr = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer_attr, 
                                        step_size = 5,
                                        gamma=0.9997)

        self.lrs_optimizer_fr = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer_fr, 
                                        step_size = 5,
                                        gamma=0.9997)


    def evaluate_attr(self, eval_dl):
        self.base_model.eval()
        self.attr_model.eval()

        preds = []
        labels = []

        loop = tqdm(total = len(eval_dl))
        for imgs, attr_vec, cls_id in eval_dl:
            imgs = imgs.to(my_device)
            attr_vec = attr_vec.to(my_device)
            cls_id = cls_id.to(my_device)

            # get global and local image features
            gl_feats, lc_feats = self.base_model(imgs) 
            pred = torch.nn.functional.sigmoid(self.attr_model(gl_feats))

            # converting in a 1D vector
            bs, num_atrr = pred.size()
            pred = pred.reshape(bs*num_atrr)
            attr_vec = attr_vec.reshape(bs*num_atrr) 

            preds += pred.data.cpu().tolist()
            labels += attr_vec.data.cpu().tolist()

            loop.update(1)
            loop.set_postfix()

        loop.close()
        print("\n calculating attribution prediction accuracy: ")
        acc, _ = calculate_acc(preds, labels, low_thresh=0.0, interval=0.01) 
        return acc 


    def train_epoch(self):
        self.base_model.train()
        self.attr_model.train()
        self.metric_fc.train()

        epoch = self.args.current_epoch 
        total_length = len(self.train_dl) * self.args.batch_size
        total_loss = 0
        total_fr_l = 0
        total_attr_l = 0

        loop = tqdm(total = len(self.train_dl))

        for imgs, attr_vec, label in self.train_dl:   
            imgs = imgs.to(my_device)
            attr_vec = attr_vec.to(my_device)
            label = label.to(my_device)
        
            gl_feats, lc_feats = self.base_model(imgs) 
            pred_attrs = self.attr_model(gl_feats)
            output = self.metric_fc(gl_feats, label)

            # zero grad
            self.optimizer_attr.zero_grad()
            self.optimizer_fr.zero_grad()
            if epoch > self.args.freeze: self.optimizer_model.zero_grad() 

            # loss
            loss_attr =  self.criterion_attr(pred_attrs, attr_vec)
            loss_fr = self.criterion_fr(output, label)
            total_loss = loss_attr + loss_fr
            total_loss.backward()

            total_attr_l += loss_attr.item()
            total_fr_l += loss_fr.item()

            # updating weights
            if epoch > self.args.freeze: self.optimizer_model.step()
            self.optimizer_attr.step()
            self.optimizer_fr.step() 

            # updating scheduler
            self.lrs_optimizer_model.step()
            self.lrs_optimizer_attr.step()
            self.lrs_optimizer_fr.step()

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{self.args.epochs}]')
            loop.set_postfix()

        loop.close()
        print(' | epoch {:3d} |' .format(self.args.current_epoch))
        print("Attribute loss: {:3.5f} ".format(total_attr_l / total_length))
        print("FR loss: {:3.5f} ".format(total_fr_l / total_length))

   
    def train(self):
        self.val_acc = 0
        for epoch in range(0, self.args.epochs):
            self.args.current_epoch = epoch

            self.train_epoch()
            
            if epoch  > self.args.valid_interval:
                print("\nLet's validate the model")
                acc = self.evaluate_attr(eval_dl = self.valid_dl)

                print("Let's evaluate FR performance")
                self.evaluate_fr()

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

    parser.add_argument('--dataset',        type=str,   default="celeba_dialog",    help='celeba|lfw|celeba_dialog')
    parser.add_argument('--batch_size',     type=int,   default=128,         help='batch size')
    parser.add_argument('--epochs',         type=int,   default=15,          help='Number of epochs')

    parser.add_argument('--architecture',   type=str,   default="ir_50",     help='iResNet Architecture 18|50|101')
    parser.add_argument('--model_type',     type=str,   default="arcface",   help='architecture of the model: arcface | cosface')
    parser.add_argument('--checkpoint_path',type=str,   default="checkpoint/fr_attr/fr_attr_arcface_14.pth",    help='path of the saved cp')
    parser.add_argument('--freeze',         type=int,   default=6,           help='Number of epoch pretrained model frezees')

    parser.add_argument('--valid_interval',     type=int,   default=5,       help='valid (epochs)')
    return  parser.parse_args(argv)


lfw_cfg = SimpleNamespace(
    num_classes = 4500
)

celeba_cfg = SimpleNamespace(
    num_classes = 4500
)

celeba_dialog_cfg = SimpleNamespace(
    num_classes = 8211
)


setup_cfg = SimpleNamespace(
    weights_arcface_18 = "./weights/arcface_ir18_ms1mv3.pth",
    weights_arcface_50 = "./weights/arcface_ir50_ms1mv3.pth", 
    weights_arcface_101 = "./weights/arcface_ir101_ms1mv3.pth",  

    metric = "arc_margin", 
    loss = "focal_loss", 
    use_se = False,
    manual_seed = 61,
    num_workers = 4,
    is_ident = False,
)


if __name__ == "__main__":
    c_args = parse_arguments(sys.argv[1:])

    if c_args.dataset == "celeba":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **celeba_cfg.__dict__) 
    
    elif c_args.dataset == "celeba_dialog":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **celeba_dialog_cfg.__dict__)

    elif c_args.dataset == "lfw":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__, **lfw_cfg.__dict__)
    

    # set seed
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.data_dir = os.path.join("./datasets", args.dataset)
    args.ann_root = os.path.join(args.data_dir, "annotation")
    args.output_dir = f'output/fr_attr_{args.dataset}'

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    t = Trainer(args)
    print("start training ...")

    if args.train == True:
        t.train()
    elif args.train == False:
        t.evaluate_attr(eval_dl = t.test_dl)
        


"""
RUN the code
python3 train_fr_attribute.py  --architecture ir_50 --dataset celeba_dialog --evaluate
"""