import torch.utils.data as data
import os, json 
import numpy as np
import numpy.random as random
from data.utils import *

################################################################
#                    Train Dataset
################################################################

class AttrDataset(data.Dataset):
    def __init__(self, split="train", args=None):
        print(f"################ Loading part: ", split)
        self.model_type = args.model_type
        self.dataset = args.dataset 
        self.split = split

        if split == "train":    ann_split="train.json" 
        elif split == "valid":  ann_split="val.json"
        elif split == "test":   ann_split="test.json" 

        ann_file = os.path.join(args.ann_root, ann_split)
        print('loading json file: ' + ann_file)
        self.annotation = json.load(open(ann_file, 'r'))

        if self.dataset == "celeba_dialog":
            attr_file = os.path.join(args.data_dir, "attribute.json")
            self.attribute = json.load(open(attr_file, 'r'))
            
        self.img_dir = os.path.join(args.data_dir, "images")


    def __getitem__(self, index):
        ann = self.annotation[index] 
        image_path = os.path.join(self.img_dir, ann['image'])
        image = transform_images(image_path, self.split)
    
        cls_id = int(ann['image_id'])

        if self.dataset == "celeba_dialog":  
            cls_id = cls_id - 1
            attr_vec = torch.as_tensor(self.attribute[ann['image']], dtype=torch.float32)

        elif self.dataset == "lfw_a":
            attr_vec = torch.as_tensor(ann['attribute'], dtype=torch.float32)
        return image, attr_vec, cls_id 


    def __len__(self):
        return len(self.annotation)


class FrTrainDataset():
    def __init__(self, args):
        print(f"######### Train Dataset: {args.dataset} ########")
        self.data_dir = args.data_dir
        self.filenames, self.class_id = get_imgs_dir(self.data_dir, args)
        self.args = args

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_name = self.filenames[index]
        cls_id = self.class_id[index]

        imgs = transform_images(img_name, "train")
        return imgs, torch.tensor(cls_id) 


class FrTestDataset:
    def __init__(self, args, test_dataset):
        self.split= "test"
        self.data_dir =  os.path.join("./datasets", test_dataset)
        test_ver_list  = os.path.join(self.data_dir, "test_pairs.txt")

        print("Loading %s dataset" % test_dataset)
        self.imgs_pair, self.pair_label = self.get_test_list(test_ver_list)

    def get_test_list(self, test_ver_list):
        with open(test_ver_list, 'r') as fd:
            pairs = fd.readlines()
        imgs_pair = []
        pair_label = []

        for pair in pairs:
            splits = pair.split(" ")
            imgs = [splits[0], splits[1]]
            imgs_pair.append(imgs)
            pair_label.append(int(splits[2]))
        return imgs_pair, pair_label


    def __getitem__(self, index):
        imgs = self.imgs_pair[index]
        pair_label = self.pair_label[index]
        data_dir = os.path.join(self.data_dir, "test")

        img1_name = imgs[0] 
        img2_name = imgs[1]

        img1_path = os.path.join(data_dir, img1_name)
        img2_path = os.path.join(data_dir, img2_name)

        img1 = transform_images(img1_path, self.split)
        img2 = transform_images(img2_path, self.split)

        img1_h = do_flip_test_images(img1_path)
        img2_h = do_flip_test_images(img2_path)

        return img1, img2, img1_h, img2_h, pair_label

    def __len__(self):
        return len(self.imgs_pair)


def get_imgs_dir(data_dir, args):
    img_dir = os.path.join(data_dir, "images")
    sub_ls = sorted(os.listdir(img_dir), key= lambda x: int(x))
    sub_ls = sub_ls

    all_labels = []
    all_files = []
    print("Subject list: %s, %s ..... %s" % (sub_ls[0], sub_ls[1], sub_ls[-1]))
    
    for sub in sub_ls:
        sub_dir = os.path.join(img_dir, sub)

        sub_imgs_dir = [os.path.join(sub_dir, img) for img in os.listdir(sub_dir)]
        all_files += sub_imgs_dir 
        all_labels += [int(sub)] * len(sub_imgs_dir) 

    print("Labels: %d, %d ..... %d" % (int(sub_ls[0]), int(sub_ls[1]), int(sub_ls[-1])))
    return all_files, all_labels