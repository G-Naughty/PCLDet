"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
# import dota_utils as util
# import re
# import time
# import polyiou
import json
from collections import defaultdict
from pycocotools.coco import COCO

## the thresh for nms when merge image
nms_thresh = 0.3


def Onlyscore(Result_path, FinalResult_path):
    with open(Result_path, "r") as fp:
        Results = json.load(fp)
    for i in range(0, len(Results)):
        Results[i]['score'] = Results[i]['score'] / 2 + 0.5
    with open(FinalResult_path, "w") as fout:
        json.dump(Results, fout, ensure_ascii=False)


def select_cat(img_result):
    cats_info = []
    # [conf,cat]paixu
    for cat_id in img_result:
        cat_results = np.array(img_result[cat_id])
        if len(cat_results) > 0:
            new_objs = cat_results[np.argsort(-cat_results[:, 0])]
            cats_info.append([new_objs[0][0], cat_id])
    cats_info = np.array(cats_info)
    cats_info = cats_info[np.argsort(-cats_info[:, 0])]
    cats_info = cats_info.tolist()
    results = [cats_info[0]]
    for i in range(1, len(cats_info)):
        if ((cats_info[0][1] - cats_info[i][1]) < 0.2):
            results.append(cats_info[i])
    return results




if __name__ == '__main__':
    # see demo for example
    # ImgDict_path='/home/disk/aliyun/multi_scale/val/annotations/ImgDict.json'
    Result_path = '/mnt/vdb/isalab205/workdir/contrastive_swinb_3x_800-1800_anchor_bs2x8_pcl/result.json'
    FinalResult_path = '/mnt/vdb/isalab205/workdir/contrastive_swinb_3x_800-1800_anchor_bs2x8_pcl/result_final.json'
    Onlyscore(Result_path, FinalResult_path)
