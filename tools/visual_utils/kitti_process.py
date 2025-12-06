import torch
import matplotlib
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

import struct
import os, time
import open3d as o3d

from tqdm import *

from ken_utils import Multi_Thread_Process, check_dir_existence, show_pc

class Kitti_Dimension_Reduction():
    """
        Written by Ken_Edward of SEU, on 10/8/2023.

        Function:
        对kitti点云数据进行预处理，使其降维为bev或fv.

        Args:None

        Usage:参见pc_calibration方法与get_calib_dat方法.
    """
    def __init__(self, **kwargs):
        pass

    def get_calib_dat(self):
        """
            function: 获得相机校正矩阵与bev-img坐标转换矩阵
        """
        # 读取矫正文件内容
        with open(self.calib_path, 'r') as f:
            calib = f.readlines()
        # P2 (3 x 4) for left eye
        self.P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
        R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
        # Add a 1 in bottom-right, reshape to 4 x 4,还原为变换矩阵
        R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
        self.R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
        # 还原平移矩阵
        Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
        self.Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)


    def pc_calibration(self, pc_path, calib_path, img):
        """
            function: 将点云并入视场内
        """
        self.pc_path = pc_path
        self.calib_path = calib_path
        self.img = img
        self.IMG_SIZE = self.img.shape

        self.get_calib_dat()
        pc = np.fromfile(self.pc_path, dtype=np.float32).reshape((-1, 4))
        self.points = pc[:, 0:3] # lidar xyz (front, left, up)
        velo = np.insert(self.points, 3, 1, axis=1).T
        velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)  # 剔除x小于0的点
        cam = self.P2.dot(self.R0_rect.dot(self.Tr_velo_to_cam.dot(velo)))# 进行平移变换

        cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)# 剔除z小于0的点

        # get u,v,z
        cam[:2] /= cam[2, :]  # 沿z轴缩放

        u, v, z = cam
        u_out = np.logical_or(u < 0, u > self.IMG_SIZE[1])
        v_out = np.logical_or(v < 0, v > self.IMG_SIZE[0])
        outlier = np.logical_or(u_out, v_out)
        self.cam = np.delete(cam, np.where(outlier), axis=1)
        # show_pc(self.cam.T)
        return self.cam

    def front_view_visual(self, **kwargs):
        u, v, z = self.cam
        plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
        plt.imshow(self.img)
        plt.scatter([u], [v], c=[z], cmap='gray', alpha=0.5, s=2)
        plt.axis([0, self.IMG_SIZE[1], self.IMG_SIZE[0], 0])
        plt.title('test')
        plt.show()

    def save_front_view(self, save_path):
        """
            function: 生成并保存fv图
        """
        u, v, z = np.array(self.cam, dtype=np.int16)
        mask = np.zeros([self.IMG_SIZE[0], self.IMG_SIZE[1]], dtype=np.int16)
        mask[v, u] = z[:]
        cv2.imwrite(save_path, mask)
        return mask

    def save_birdeye_view(self, pc_path, save_path):
        """
            function: 生成并保存bev图
        """
        pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
        self.points = pc[:, 0:3]  # lidar xyz (front, left, up)
        for n in range(3):
            self.points[:,n] = self.points[:,n]-self.points[:,n].min()
            self.points[:, n] = self.points[:,n]/0.01
        self.points = np.array(self.points, dtype=np.int16)
        # print(self.points[:,1].max())
        mask = np.zeros([self.points[:,0].max()+1, self.points[:,1].max()+1], dtype=np.int16)
        mask[self.points[:,0], self.points[:,1]] = self.points[:,2]
        cv2.imwrite(save_path, mask)
        # return mask

    def save_front_view_1(self, save_path, show_img=False, **kwargs):
        """
            function: 保存fv图
        """
        plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
        plt.axis([0, self.IMG_SIZE[1], self.IMG_SIZE[0], 0])
        plt.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        u, v, z = np.array(self.cam, dtype=np.int16)
        mask = np.zeros([self.IMG_SIZE[0], self.IMG_SIZE[1]], dtype=np.int16)
        mask[v, u] = z[:]
        plt.imshow(mask, cmap='gray')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        if show_img:
            plt.imshow(mask, cmap='gray')
            plt.show()
        self.mask = mask
        plt.close()
        return mask

    def augment_2D(self, show_img=False, **kwargs):
        """
            function: 进行fv图的数据增强
        """
        kernel = np.ones((3, 5), np.uint8)
        fv = cv2.filter2D(self.mask, -1, kernel)/3
        if show_img:
            plt.imshow(fv, cmap='gray')
            plt.show()

def gfv_test():
    pc_dir = './data_object_velodyne/testing/velodyne/000000.bin'
    calib_dir = './data_object_calib/testing/calib/000000.txt'
    img_dir = './data_object_image_2/testing/image_2/000000.png'

    img = cv2.imread(img_dir)
    gfv = Kitti_Dimension_Reduction()
    gfv.get_real_front_view(pc_dir, calib_dir, img)
    # gfv.front_view_visual()
    gfv.save_front_view(save_path='1.jpg', show_img=True)
    gfv.fv_augment(show_img=True)

def multi_thread_sub_kitti_gfv_process(img_list):
    pc_dir = './data_object_velodyne/testing/velodyne/'
    calib_dir = './data_object_calib/testing/calib/'
    img_dir = './data_object_image_2/testing/image_2/'
    save_fv_dir = './data_object_image_2/testing/fv/'
    s = time.time()
    gfv = Kitti_Dimension_Reduction()
    for img_name in img_list:
        print(img_name)
        img_index = img_name.split('.')[0]

        img_path = img_dir + img_name
        pc_path = pc_dir + img_index + '.bin'
        calib_path = calib_dir + img_index + '.txt'
        save_fv_path = save_fv_dir + img_index + '.png'

        img = cv2.imread(img_path)
        gfv.pc_calibration(pc_path, calib_path, img)
        print('processing time:', time.time()-s)
        gfv.save_front_view(save_path=save_fv_path)

def kitti_gfv_process():
    pc_dir = './data_object_velodyne/testing/velodyne/'
    calib_dir = './data_object_calib/testing/calib/'
    img_dir = './data_object_image_2/testing/image_2/'
    save_fv_dir = './data_object_image_2/testing/fv/'
    check_dir_existence(save_fv_dir)

    img_list = os.listdir(img_dir)
    # img_list_len = len(img_list)
    # print(img_list_len)

    mtp_kitti_gfv_process = \
        Multi_Thread_Process(func=multi_thread_sub_kitti_gfv_process,
                            do_list=img_list,
                             thread_num=10,)
    mtp_kitti_gfv_process.init_thread()
    mtp_kitti_gfv_process.run()

def gbev_test():
    pc_dir = './data_object_velodyne/testing/velodyne/000007.bin'
    calib_dir = './data_object_calib/testing/calib/000007.txt'
    img_dir = './data_object_image_2/testing/image_2/000007.png'

    img = cv2.imread(img_dir)
    g2dv = Kitti_Dimension_Reduction()
    g2dv.pc_calibration(pc_dir, calib_dir, img)
    # gfv.front_view_visual()
    g2dv.save_birdeye_view(save_path='1.jpg')
    # gfv.fv_augment(show_img=True)

def multi_thread_sub_kitti_gbev_process(img_list):
    pc_dir = './data_object_velodyne/training/velodyne/'
    calib_dir = './data_object_calib/training/calib/'
    img_dir = './data_object_image_2/training/image_2/'
    save_bev_dir = './data_object_image_2/training/bev/'
    gbev = Kitti_Dimension_Reduction()
    # print('\n')
    for img_name in img_list:
        # print(img_name)
        img_index = img_name.split('.')[0]

        img_path = img_dir + img_name
        pc_path = pc_dir + img_index + '.bin'
        calib_path = calib_dir + img_index + '.txt'
        save_bev_path = save_bev_dir + img_index + '.png'

        img = cv2.imread(img_path)
        # gfv.pc_calibration(pc_path, calib_path, img)
        gbev.save_birdeye_view(pc_path=pc_path, save_path=save_bev_path)
        print('\b')

def kitti_gbev_process():

    pc_dir = './data_object_velodyne/training/velodyne/'
    calib_dir = './data_object_calib/training/calib/'
    img_dir = './data_object_image_2/training/image_2/'
    save_fv_dir = './data_object_image_2/training/bev/'
    check_dir_existence(save_fv_dir)

    img_list = os.listdir(img_dir)
    # img_list_len = len(img_list)
    # print(img_list_len)

    mtp_kitti_gfv_process = \
        Multi_Thread_Process(func=multi_thread_sub_kitti_gbev_process,
                            do_list=img_list,
                             thread_num=12,)
    mtp_kitti_gfv_process.init_thread()
    mtp_kitti_gfv_process.run()

if __name__ == '__main__':
    kitti_gfv_process()
    # gbev_test()
    pass
