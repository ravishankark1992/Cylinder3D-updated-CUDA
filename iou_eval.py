import pandas as pd
import numpy as np
from collections import Counter
from argparse import ArgumentParser
import yaml
import os
from glob import glob
from tqdm import tqdm

moving_remap = {
    252: 10,
    253: 11,
    254: 30,
    255: 32,
    256: 16,
    257: 13,
    258: 18,
    259: 20
}

class EvalIou:
    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        pass

    def compute_iou(self, pred, gt):
        # IOU formula
        # intersection of class ids / Union of class ids
        intersection = np.logical_and(pred, gt)
        union = np.logical_or(pred, gt)
        iou = np.sum(intersection) / np.sum(union)
        print("intersection, union", Counter(intersection), Counter(union))
        print("iou: ", iou)
        return iou
    
def iou_calculate(path_to_pred, path_to_gt, config):
    if path_to_pred.split('/')[-1] != path_to_gt.split('/')[-1]:
        print(path_to_pred, path_to_gt)
    pred = np.fromfile(path_to_pred, dtype=np.uint16)
    pred = pred.reshape((-1))
    gt=np.fromfile(path_to_gt, dtype=np.uint16)
    gt=gt.reshape((-1))
    reshaped_pred = pred.reshape(int(len(pred)/2), 2)
    reshaped_gt = gt.reshape(int(len(gt)/2), 2)
    labels = config["labels"]
    # remap moving labels
    y=reshaped_gt[:, 0]
    for key in moving_remap.keys():
        reshaped_gt[:, 0][reshaped_gt[:, 0] == key] = moving_remap[key]
    pred_counter = Counter(reshaped_pred[:, 0])
    gt_counter = Counter(reshaped_gt[:, 0])
    pred_label_values = {}
    gt_label_values = {}
    # Display with labels

    for item in labels:
        if item in pred_counter.keys():
            pred_label_values[labels[item]] = pred_counter[item]
        if item in gt_counter.keys():
            gt_label_values[labels[item]] = gt_counter[item]
    print("prediction: ", pred_label_values)
    print("GT: ", gt_label_values)
    import pdb; pdb.set_trace()
    return reshaped_pred, reshaped_gt
    

def main(path_to_pred, path_to_gt, config_path):
    config_path = os.path.join(config_path)
    eval = EvalIou(config_path)
    score_list = []
    folder_score_list = []
    # loop
    pred_files = None
    #reshaped_pred, reshaped_gt = iou_calculate(pred_files[2], gt_files[2], eval.config)
    folder_list = ['0'+str(i) for i in range(0,10)]
    folder_list.extend([str(i) for i in range(10,11)])
    for folder in tqdm(folder_list):
        pred_files = sorted(glob("/".join([path_to_pred, folder, "*.label"])))
        gt_files = sorted(glob("/".join([path_to_gt, folder, "labels", "*.label"])))
        import pdb
        for idx, _ in enumerate(path_to_gt):
            reshaped_pred, reshaped_gt = iou_calculate(pred_files[idx], gt_files[idx], eval.config)
            score = eval.compute_iou(reshaped_pred[:, 0], reshaped_gt[:, 0])
            score_list.append(score)
        folder_score=np.mean(score_list)
        folder_score_list.append(folder_score)
    print("each folder score: ", folder_score_list)
    print("overall_score: ", np.mean(folder_score_list))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--config_path', type=str, default="config/label_mapping/semantic-kitti.yaml")
    args = parser.parse_args()
    pred = args.pred
    gt = args.gt
    config_path = args.config_path
    main(pred, gt, config_path)