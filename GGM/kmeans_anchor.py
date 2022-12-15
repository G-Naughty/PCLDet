
import os
import numpy as np
from pycocotools.coco import COCO
import copy

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, ratios=[0.25,0.5,1,2,4],scale=6,base_sizes=[4,8,16,32,64], dist=np.median): #np.median
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    k = len(base_sizes)*len(ratios)
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()
    #clusters = boxes[np.random.choice(rows, 5, replace=False)]
    # the Forgy method will fail if the whole array contains the same rows
    h_ratios = np.sqrt(ratios).reshape(1, -1)
    w_ratios = (1 / h_ratios).reshape(1, -1)
    base_sizes = np.array(base_sizes).reshape(-1, 1)
    ws = (base_sizes * w_ratios*scale).reshape(-1)
    hs = (base_sizes * h_ratios * scale).reshape(-1)
    clusters = np.array([[ws[i], hs[i]]for i in range(k)])
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            ori = copy.deepcopy(clusters[cluster])
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
            if (np.isnan(clusters[cluster]).any()):
                clusters[cluster] = copy.deepcopy(ori)

        last_clusters = nearest_clusters

    return clusters


if __name__ == '__main__':
    annFile = r'/home/disk/aliyun/train/annotations/instances_train2017.json'
    coco = COCO(annFile)
    cats_info=[]
    for id in coco.cats:

        annIds=coco.getAnnIds(catIds=id)
        annList = coco.loadAnns(annIds)
        cat_info=[]
        for ann in annList:
            w_h = np.array([ann['bbox'][2], ann['bbox'][3]])
            cat_info.append(w_h)
        cats_info+=cat_info
    #cats_info = np.concatenate(cats_info, axis=0)

    cats_info = np.asarray(cats_info)

    k = kmeans(cats_info)
    print(k)
    area_new = []
    ratio_new = []
    for abbox in k:
        ratio_new.append(abbox[0] / abbox[1])
        area_new.append(abbox[0] * abbox[1])
    print('ratios:')
    print(np.sort(ratio_new))
    base_sizes = [4, 8, 16, 32, 64]
    scales = []
    for i in range(len(area_new)):
        t = int(i / 5)
        #print('base_size=' + str(base_sizes[t]))
        scales.append(np.sqrt(area_new[i] / (base_sizes[t] * base_sizes[t])))
    print('scales')
    print(np.sort(scales))
    print("Accuracy: {:.2f}%".format(avg_iou(cats_info, k) * 100))
    print("Boxes:\n {}".format(k))

    ratios = np.around(k[:, 0] / k[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))