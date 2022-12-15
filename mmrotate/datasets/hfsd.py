# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial
import shutil
import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import obb2poly_np, poly2obb_np
from mmrotate.core.evaluation import eval_rbbox_map
from .builder import ROTATED_DATASETS
import pickle
import copy




@ROTATED_DATASETS.register_module()
class HFSDDataset(CustomDataset):
    """HFSD dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('Nimitz_Aircraft_Carrier', 'Barracks_Ship', 'Container_Ship', 'Fishing_Vessel',
               'Henry_J._Kaiser-class_replenishment_oiler', 'Other_Warship', 'Yacht',
               'Freedom-class_littoral_combat_ship', 'Arleigh_Burke-class_Destroyer',
               'Lewis_and_Clark-class_dry_cargo_ship', 'Towing_vessel', 'unknown',
               'Powhatan-class_tugboat', 'Barge', '055-destroyer', '052D-destroyer',
               'USNS_Bob_Hope', 'USNS_Montford_Point', 'Bunker', 'Ticonderoga-class_cruiser',
               'Oliver_Hazard_Perry-class_frigate', 'Sacramento-class_fast_combat_support_ship',
               'Submarine', 'Emory_S._Land-class_submarine_tender', 'Hatakaze-class_destroyer',
               'Murasame-class_destroyer', 'Whidbey_Island-class_dock_landing_ship',
               'Hiuchi-class_auxiliary_multi-purpose_support_ship', 'USNS_Spearhead',
               'Hyuga-class_helicopter_destroyer', 'Akizuki-class_destroyer', 'Bulk_carrier',
               'Kongō-class_destroyer', 'Northampton-class_tug', 'Sand_Carrier',
               'Iowa-class_battle_ship', 'Independence-class_littoral_combat_ship',
               'Tarawa-class_amphibious_assault_ship', 'Cyclone-class_patrol_ship',
               'Wasp-class_amphibious_assault_ship', '074-landing_ship', '056-corvette',
               '721-transport_boat', '037Ⅱ-missile_boat', 'Traffic_boat', '037-submarine_chaser',
               'unknown_auxiliary_ship', '072Ⅲ-landing_ship', '636-hydrographic_survey_ship',
               '272-icebreaker', '529-Minesweeper', '053H2G-frigate', '909A-experimental_ship',
               '909-experimental_ship', '037-hospital_ship', 'Tuzhong_Class_Salvage_Tug',
               '022-missile_boat', '051-destroyer', '054A-frigate', '082Ⅱ-Minesweeper',
               '053H1G-frigate', 'Tank_ship', 'Hatsuyuki-class_destroyer',
               'Sugashima-class_minesweepers', 'YG-203_class_yard_gasoline_oiler',
               'Hayabusa-class_guided-missile_patrol_boats', 'JS_Chihaya',
               'Kurobe-class_training_support_ship', 'Abukuma-class_destroyer_escort',
               'Uwajima-class_minesweepers', 'Osumi-class_landing_ship',
               'Hibiki-class_ocean_surveillance_ships', 'JMSDF_LCU-2001_class_utility_landing_crafts',
               'Asagiri-class_Destroyer', 'Uraga-class_Minesweeper_Tender', 'Tenryu-class_training_support_ship',
               'YW-17_Class_Yard_Water', 'Izumo-class_helicopter_destroyer', 'Towada-class_replenishment_oilers',
               'Takanami-class_destroyer', 'YO-25_class_yard_oiler', '891A-training_ship', '053H3-frigate',
               '922A-Salvage_lifeboat', '680-training_ship', '679-training_ship', '072A-landing_ship',
               '072Ⅱ-landing_ship', 'Mashu-class_replenishment_oilers', '903A-replenishment_ship',
               '815A-spy_ship', '901-fast_combat_support_ship', 'Xu_Xiake_barracks_ship',
               'San_Antonio-class_amphibious_transport_dock', '908-replenishment_ship', '052B-destroyer',
               '904-general_stores_issue_ship', '051B-destroyer', '925-Ocean_salvage_lifeboat',
               '904B-general_stores_issue_ship', '625C-Oceanographic_Survey_Ship', '071-amphibious_transport_dock',
               '052C-destroyer', '635-hydrographic_Survey_Ship', '926-submarine_support_ship', '917-lifeboat',
               'Mercy-class_hospital_ship', 'Lewis_B._Puller-class_expeditionary_mobile_base_ship',
               'Avenger-class_mine_countermeasures_ship', 'Zumwalt-class_destroyer', '920-hospital_ship',
               '052-destroyer', '054-frigate', '051C-destroyer', '903-replenishment_ship', '073-landing_ship',
               '074A-landing_ship', 'North_Transfer_990', '001-aircraft_carrier', '905-replenishment_ship',
               'Hatsushima-class_minesweeper', 'Forrestal-class_Aircraft_Carrier', 'Kitty_Hawk_class_aircraft_carrier',
               'Blue_Ridge_class_command_ship', '081-Minesweeper', '648-submarine_repair_ship',
               '639A-Hydroacoustic_measuring_ship', 'JS_Kurihama', 'JS_Suma', 'Futami-class_hydro-graphic_survey_ships',
               'Yaeyama-class_minesweeper', '815-spy_ship', 'Sovremenny-class_destroyer')

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(HFSDDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Params:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.bmp')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.bmp'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.bmp'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['labels'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Params:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Params:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)
        new_classes_name=[]
        for cls in self.CLASSES:
            new_cls_name = cls.strip().split('_')
            new_cls_name = ' '.join(new_cls_name)
            new_classes_name.append(new_cls_name)
        files = [
            osp.join(out_folder, cls + '.txt')
            for cls in new_classes_name
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    img_name = img_id.strip().split('__1024__0___0')[0]+'.bmp'
                    txt_element = [img_name, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()
        # target_name = osp.split(out_folder)[-1]
        # with zipfile.ZipFile(
        #         osp.join(out_folder, target_name + '.zip'), 'w',
        #         zipfile.ZIP_DEFLATED) as t:
        #     for f in files:
        #         t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
            files.
                If not specified, a temp folder will be created. Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        # print('\nMerging patch bboxes into full image!!!')
        # start_time = time.time()
        # id_list, dets_list = self.merge_det(results, nproc)
        # stop_time = time.time()
        # print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(self.img_ids, results,
                                                submission_dir)
        # mmcv.dump(results, submission_dir+'dets.pkl')
        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Params:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results
