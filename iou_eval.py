import pandas as pd
import numpy as np
from collections import Counter
from argparse import ArgumentParser
import yaml
import os

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
        return iou
    


def main(path_to_pred, path_to_gt, config_path):
    config_path = os.path.join(config_path)
    eval = EvalIou(config_path)
    pred = np.fromfile(path_to_pred, dtype=np.uint16)
    pred = pred.reshape((-1))
    gt=np.fromfile(path_to_gt, dtype=np.uint16)
    gt=gt.reshape((-1))
    reshaped_pred = pred.reshape(int(len(pred)/2), 2)
    reshaped_gt = gt.reshape(int(len(gt)/2), 2)
    labels = eval.config["labels"]
    
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
    score = eval.compute_iou(reshaped_pred[:, 0], reshaped_gt[:, 0])
    print(score)

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