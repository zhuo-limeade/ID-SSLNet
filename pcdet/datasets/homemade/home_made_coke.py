import copy
import pickle
import sys 
import numpy as np

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils, object3d_kitti
from pcdet.datasets.dataset import DatasetTemplate


class Homemade_Dataset_COKE_3cls(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.cls_dict = {'Worker':1,'Car':2,'Sign':3}
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'bin' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label' / ('%s.txt' % idx)
        assert label_file.exists()
        labels = {
            'name':[],
            'truncated':[],
            'occluded':[],
            'alpha':[],
            'bbox':[],
            'dimensions':[],
            'location':[],
            'rotation_y':[],
            'score':[],
            'difficulty':[],
        }
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                line[-1] = line[-1].split('\n')[0]
                labels['name'].append(line[0])
                labels['truncated'].append(np.array(line[1], dtype=np.float32))
                labels['occluded'].append(np.array(line[2], dtype=np.float32))
                labels['alpha'].append(np.array(line[3], dtype=np.float32))
                labels['bbox'].append(np.array(line[4:8], dtype=np.float32))
                labels['dimensions'].append(np.array(line[11:14], dtype=np.float32))
                labels['location'].append(np.array(line[8:11], dtype=np.float32))
                labels['rotation_y'].append(np.array(line[14], dtype=np.float32))
                labels['score'].append(-1)
                labels['difficulty'].append(1)
            for n in labels:
                labels[n] = np.array(labels[n])
        return labels

    def get_infos(self, num_workers=1, has_label=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = self.get_label(sample_idx)
                num_objects = len(annotations['name'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                try :
                    l, h, w = dims[:, 2:3], dims[:, 1:2], dims[:, 0:1]
                except:
                    pass
                annotations['dimensions'][:num_objects] = np.concatenate([l,h,w], axis=1)
                # loc[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc, l,w,h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('ssl_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval_homemade_coke as eval_homemade_1

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        PR_detail_dict = {}
        ap_result_str, ap_dict = eval_homemade_1.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names, PR_detail_dict)
        pr_dat = PR_detail_dict['3d']
        # try:
        #     f = open('/home/ken/program/dataset/result_process/KITTI_VIZ_3D/val_dict/second.txt', 'w+')
        #     for cls in range(3):
        #         for point in range(41):
        #             line = ''
        #             for dif in range(3): 
        #                 line += str(pr_dat[cls, dif, 0, point])
        #                 line += ' '
        #             line += '\n'
        #             f.write(line)
        #     f.close()
        #     print('write PR data successed ...')
        # except: 
        #     f = open('/home/ken/program/dataset/result_process/KITTI_VIZ_3D/val_dict/second.txt', 'w+')
        #     f.write("write PR data failed!")
        #     print("write PR data failed!")

        return ap_result_str, ap_dict

    # def get_lidar_box_anno(self, ):
    #     -(r + np.pi / 2)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            # gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = Homemade_Dataset_COKE_3cls(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('ssl_infos_%s.pkl' % train_split)
    val_filename = save_path / ('ssl_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'ssl_infos_trainval.pkl'
    test_filename = save_path / 'ssl_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    ssl_infos_train = dataset.get_infos(num_workers=workers, has_label=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(ssl_infos_train, f)
    print('SSL info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    ssl_infos_val = dataset.get_infos(num_workers=workers, has_label=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(ssl_infos_val, f)
    print('SSL info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(ssl_infos_train + ssl_infos_val, f)
    print('SSL info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(kitti_infos_test, f)
    # print('SSL info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    
    import sys

    import yaml
    from pathlib import Path
    from easydict import EasyDict
    dataset_cfg = EasyDict(yaml.safe_load(open('/home/ken/workspace/OpenPCDet/tools/cfgs/homemade/home_made_coke_3cls.yaml')))
    ROOT_DIR = '/media/ken/backup_u/dataset/coke1/'
    ROOT_DIR = Path(ROOT_DIR).resolve()
    create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Worker','Car','Sign'],
        data_path=ROOT_DIR,
        save_path=ROOT_DIR,
        workers=1
    )
        
    # dataset = SSL_Homemade_Dataset(dataset_cfg=dataset_cfg, class_names=['roundabout'],root_path=ROOT_DIR, training=False)
    # dataset.set_split('train')
    # ssl_infos_train = dataset.get_infos(num_workers=2, has_label=True)
    