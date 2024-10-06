import sys
import os.path as osp
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.nn import functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models import arcface 

#ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
#sys.path.insert(0, ROOT_PATH)
#from utils.attribute import caps_all_attributes


def KFold(n, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        frac = int(n/n_folds)
        test = base[i * frac : (i + 1) * frac]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1.0 if float(d[0]) > threshold else 0.0
        y_predict.append(same)
        y_true.append(float(d[1])) #int
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def get_tpr(fprs, tprs):
    fpr_val = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3]
    tpr_fpr_row = []
    for fpr_iter in np.arange(len(fpr_val)):
        _, min_index = min(
            list(zip(abs(fprs - fpr_val[fpr_iter]), range(len(fprs)))))
        tpr_fpr_row.append(tprs[min_index] * 100)
    return tpr_fpr_row



################## scores ####################
def calculate_scores(y_score, y_true, args):
    # sklearn always takes (y_true, y_pred)
    fprs, tprs, threshold = metrics.roc_curve(y_true, y_score)
 
    fprs = np.flipud(fprs)
    tprs = np.flipud(tprs)

    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    auc = metrics.auc(fprs, tprs)
    tpr_fpr_row = get_tpr(fprs, tprs)
    total_score = tpr_fpr_row[0] + tpr_fpr_row[1] + tpr_fpr_row[2] 

    print("AUC {:.4f} | EER {:.4f} | TPR@FPR=1e-6 {:.4f} | TPR@FPR=1e-5 {:.4f} | TPR@FPR=1e-4 {:.4f} | score {:.4f}".
        format(auc, eer, tpr_fpr_row[0], tpr_fpr_row[1], tpr_fpr_row[2], total_score))   #| TPR@FPR=1e-3 {:.4f},  tpr_fpr_row[3]

    """
        filename = os.path.join(".", args.roc_file + '.npy')
        print("saving npy file in :", filename)
        with open(filename, 'wb') as f:
            np.save(f, y_true)
            np.save(f, y_score)
    """


def calculate_acc(preds, labels, low_thresh=-1.0, interval=0.005):
    predicts = []
    num_imgs = len(preds)
    with torch.no_grad():
        for i in range(num_imgs):

            #distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\n'.format(preds[i], labels[i]))
    
    accuracy = []
    thd = []
    folds = KFold(n=num_imgs, n_folds=10)
    thresholds = np.arange(low_thresh, 1.0, interval)
    predicts = np.array(list(map(lambda line: line.strip('\n').split(), predicts)))

    print(len(predicts))
    print("starting fold ....")
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
        print("finding fold: ", idx)

    print('ACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), 
                                np.std(accuracy), np.mean(thd)))
    return np.mean(accuracy), predicts


### model for ArcFace
def prepare_arcface(args, train_mode):
    device = args.device
    if args.architecture == "ir_18":
        model = arcface.iresnet18(pretrained=False, progress=True)
        checkpoint = torch.load(args.weights_arcface_18, weights_only=True)

    elif args.architecture == "ir_50":
        model = arcface.iresnet50(pretrained=False, progress=True)
        checkpoint = torch.load(args.weights_arcface_50, weights_only=True)

    elif args.architecture == "ir_101":
        model = arcface.iresnet101(pretrained=False, progress=True)
        checkpoint = torch.load(args.weights_arcface_101, weights_only=True)
    
    model.load_state_dict(checkpoint)

    if train_mode == "my_own":
        state_dict = torch.load(args.backend_path)
        model.load_state_dict(state_dict['image_encoder'])
        for p in model.parameters():
            p.requires_grad = False

    elif train_mode == "fixed":
        for p in model.parameters():
            p.requires_grad = False 

    elif train_mode == "finetune":
        print("******* ArcFace model will be finetuned **********")
        pass 

    model.to(device)
    model.eval()
    print("loading pretrained adaface model:  ", args.architecture)
    return model 


def attribute_analysis(img, img_attr, attr_vec, text_attr):
    plt.imshow(img.detach().cpu().permute(1, 2, 0))
    pred_attributes = img_attr.detach().cpu().tolist()
    #print("predicted attributes: ", pred_attributes)

    print("for predicted image attributes")
    pred_attributes = [i if i > 0.35 else 0.0 for i in pred_attributes]
    for_print = [all_attributes[i] + " : " + str(prob) for i, prob in enumerate(pred_attributes) if prob != 0.0]
    print(for_print)

    print("for predicted text attributes")
    pred_text_attributes = text_attr.detach().cpu().tolist()
    pred_text_attributes = [i if i > 0.35 else 0.0 for i in pred_text_attributes]
    for_text_print = [caps_all_attributes[i] + " : " + str(prob) for i, prob in enumerate(pred_text_attributes) if prob != 0.0]
    print(for_text_print)
    plt.show()
    print("\n\n\n")


def test(test_dl, model, image_head, image_text_attr, 
         fusion_net, text_encoder, text_head, args):
    
    device = args.device
    fusion_net.eval()
    model.eval() 
    image_head.eval()
    text_encoder.eval()
    text_head.eval()
    image_text_attr.eval()
    preds = []
    labels = []

    loop = tqdm(total=len(test_dl))
    with torch.no_grad():
        for step, data in enumerate(test_dl, 0):
            img1, img1_h, img2, img2_h, caption1, caption2, mask1, mask2, attr_vec1, attr_vec2, pair_label = data

            img1 = img1.to(device) 
            img2 = img2.to(device) 
            pair_label = pair_label.to(device)

            # get global and local image features from face encoder
            if args.model_type == "arcface":
                gl_img1,  local_feat1 = model(img1)
                gl_img2,  local_feat2 = model(img2)

                gl_img1_h,  local_feat1_h = model(img1)
                gl_img2_h,  local_feat2_h = model(img2)


            global_feat1,  local_feat1 = image_head(gl_img1,  local_feat1)
            global_feat2,  local_feat2 = image_head(gl_img2,  local_feat2)

            global_feat1_h,  local_feat1_h = image_head(gl_img1_h,  local_feat1_h)
            global_feat2_h,  local_feat2_h = image_head(gl_img2_h,  local_feat2_h)

            # get word and caption features from text encoder
            words_emb1, sent_emb1 = encode_Bert_tokens(text_encoder, text_head, caption1, mask1, args)
            words_emb2, sent_emb2 = encode_Bert_tokens(text_encoder, text_head, caption2, mask2, args) 


            if args.printing_attr == True:
                attr_vec1 = attr_vec1.to(device)
                attr_vec2 = attr_vec2.to(device)

                img_attr1 = torch.sigmoid(image_text_attr(global_feat1))
                img_attr2 = torch.sigmoid(image_text_attr(global_feat2)) 

                #text_attr1 = torch.sigmoid(image_text_attr(sent_emb1))
                #text_attr2 = torch.sigmoid(image_text_attr(sent_emb2))

                #attribute_analysis(img1[0], img_attr1[0], attr_vec1[0], text_attr1[0])
                #attribute_analysis(img2[0], img_attr2[0], attr_vec2[0], text_attr2[0])

            # sentence & word featurs 
            out1 = fusion_net(local_feat1, words_emb1,  global_feat1, sent_emb1, gl_img1)
            out2 = fusion_net(local_feat2, words_emb2,  global_feat2, sent_emb2, gl_img2)

            out1_h = fusion_net(local_feat1_h, words_emb1,  global_feat1_h, sent_emb1, gl_img1_h)
            out2_h = fusion_net(local_feat2_h, words_emb2,  global_feat2_h, sent_emb2, gl_img2_h)
            del local_feat1, local_feat2, words_emb1, words_emb2

            out1 = torch.add(out1, out1_h) / 2.0
            out2 = torch.add(out2, out2_h) / 2.0

            cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
            pred = cosine_sim(out1, out2)
            preds += pred.data.cpu().tolist()
            labels += pair_label.data.cpu().tolist()

            # update loop information
            loop.update(1)
            loop.set_postfix()

    loop.close()

    calculate_scores(preds, labels, args)
    if args.is_ident == True:
        calculate_acc(preds, labels, args)