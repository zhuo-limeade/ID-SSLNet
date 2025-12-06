import torch
import matplotlib
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

import struct
import os
import open3d as o3d

class Draw_BEV():
    
    def __init__(self):
        self._class_list = {}
        self._colormap = [matplotlib.cm.get_cmap(name=i) for i in ['Set3']]
        self.max_class = 0

    def generate_bev(self, 
                        pc_data=None,
                        b_boxs=None,
                        box_labels=None,
                        scaling=1,
                        pc_res=0.01,
                        point_size=1,
                        show_bev_mode='img',
                        box_thickness=3,
                        ):
        """
        Written by Ken_Edward of SEU, on 15/7/2023.

        function:从PC生成BEV

        Args:
            pc_data: 点云数据;数据为2 axis,axis=0为点云中点的数量,
                    axis=1前三通道必须为(x,y,z...),
                    数据类型可为Tensor实例或Ndarray实例.
            b_boxs: 预测后的边界框;数据为2 axis,axis=0为边界框的数量,
                    axis=1通道必须为(x,y,z,w,h,l,theta),
                    数据类型可为Tensor实例或Ndarray实例.
            box_labels: 预测后的边界框对应的标签;元素为str类型,数据为2 axis,
                    axis=0为标签的数量,axis=1通道数据为str类型,
                    数据类型可为Tensor实例或Ndarray实例.
            scaling: BEV的缩放比例,默认为1;
                    但是会卡机,建议大小4以上;
                    数据类型int,可为float.
            pc_res: PC点的分辨率,转化为毫米,float型.
            point_size: 原始PC生成的BEV中点的膨胀倍数;
                    默认为1(1+1),建议为1;若为None,则不进行膨胀;
                    数据类型为int.
            show_bev_mode: BEV显示的方式,数据类型str;
                    默认为'img',显示单张图片并无限等待;若为'video'则会显示视频流.
            box_thickness: 检测边界框宽度,默认为3;数据类型int.

        Returns:
            bev: 二维BEV,结构为标准3 axis BGR图像;数据类型为Ndarray.
        """
        # 数据类型转换为Ndarray
        if isinstance(pc_data, torch.Tensor):
            pc_data = pc_data.cpu().numpy()
        if isinstance(b_boxs, torch.Tensor):
            b_boxs = b_boxs.cpu().numpy()
        if isinstance(box_labels, torch.Tensor):
            box_labels = box_labels.cpu().numpy()
        if pc_data is None:
            raise ValueError("pc_data shouldn't be None.")
        # 生成颜色dict
        if box_labels is not None:
            # print(box_labels)
            for itm in box_labels:
                if itm not in self._class_list:
                    # print(self.max_class//20)
                    color = np.array(self._colormap[self.max_class//10]((self.max_class-self.max_class//10)/10))
                    # print(type(color))
                    # color = tuple([int(i) for i in color[:3]])
                    color = color[:3]
                    self._class_list[itm] = color
                    self.max_class+=1
                    if self.max_class >=10:
                        raise ValueError("The total data classes should be less than 10.")
        
        # x = points[:,0]
        # points[:,0] = points[:,1]
        # points[:,1] = x
        # 生成BEV
        x_min, y_min, z_min = [np.min(pc_data[:,i]) for i in range(3)]
        x_max, y_max, z_max = [np.max(pc_data[:,i]) for i in range(3)]
        print(x_min,x_max)
        bev = self._point_cloud_2_birdseye(pc_data[:,:3],
                                    res = pc_res,
                                    side_range=(x_min,x_max),
                                    fwd_range=(y_min,y_max),
                                    height_range=(z_min,z_max)                                    
                                    )
        # print(np.shape(bev), bev[:,0])
        # plt.imshow(bev)
        # plt.show()
        if point_size is not None:
            kernel = np.ones((3,3),np.uint8)
            bev = cv2.dilate(bev, kernel, iterations = point_size)
        # bev = self._convert_pc_to_bev(point_size=point_size)
        bev_x,bev_y = np.shape(bev)
        img = np.zeros((bev_x,bev_y,3))
        for i in range(3):
            img[:,:,i] = bev
        # 添加检测bboxs
        if b_boxs is not None:
            for n, box in enumerate(b_boxs):
                x,y,_,w,h,_,theta = box
                box = cv2.RotatedRect(((x-x_min)/pc_res,(-y+y_max)/pc_res), 
                                        (w/pc_res,h/pc_res), 
                                        -theta*180/3.1416).points()
                # print(np.shape([box]))
                box = np.array(box,dtype='int32')
                # print(box)
                # print(theta*180/3.1416)
                if box_labels is not None:
                    # print('a')
                    # print(self._class_list[box_labels[n]])
                    # print(type(self._class_list[box_labels[n]][0]))
                    img = cv2.polylines(img, [box], color=self._class_list[box_labels[n]], isClosed=True, thickness=box_thickness)
                else:
                    img = cv2.polylines(img, [box], color=(0,255,0), isClosed=True, thickness=box_thickness)
                # cv2.imshow('bev', cv2.resize(img,(int(bev_y/6),int(bev_x/6))))
                # cv2.waitKey(0)
        img = cv2.resize(img,(int(bev_y/scaling),int(bev_x/scaling)))
        # print(np.shape(img))
        # 显示图片
        if show_bev_mode == 'img':
            cv2.imshow('bev', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif show_bev_mode == 'video':
            cv2.imshow('bev', img)
            cv2.waitKey(1)
        elif show_bev_mode == 'mpl':
            plt.imshow(img)
            plt.show()
        return img
        
        # plt.imshow(img)
        # plt.show()
        # print([float(x) for x in bev[0,:]])

    def _scale_to_255(self, a, min, max, dtype=np.uint8):
        """ 
        function: 正则化至0-255范围
        """
        return (((a - min) / float(max - min)) * 255).astype(dtype)
    
    def _point_cloud_2_birdseye(self, 
                                points,
                                res=0.01,
                                side_range=(-10., 10.),  # left-most to right-most
                                fwd_range = (-10., 10.), # back-most to forward-most
                                height_range=(-20., 20.),  # bottom-most to upper-most
                                ):
        """ 
        function: 生成BEV图像
        """
        # EXTRACT THE POINTS FOR EACH AXIS
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]

        # FILTER - To return only indices of points within desired cube
        # Three filters for: Front-to-back, side-to-side, and height ranges
        # Note left side is positive y axis in LIDAR coordinates
        f_filt = np.logical_and((x_points > side_range[0]), (x_points < side_range[1]))
        s_filt = np.logical_and((y_points > fwd_range[0]), (y_points < fwd_range[1]))
        filter = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filter).flatten()

        # KEEPERS
        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (x_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = -(y_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.ceil(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = np.clip(a=z_points,
                            a_min=height_range[0],
                            a_max=height_range[1])

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        pixel_values = self._scale_to_255(pixel_values,
                                    min=height_range[0],
                                    max=height_range[1])

        # INITIALIZE EMPTY ARRAY - of the dimensions we want
        x_max = 1 + int((side_range[1] - side_range[0]) / res)
        y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
        im = np.zeros([y_max, x_max], dtype=np.uint8)

        # FILL PIXEL VALUES IN IMAGE ARRAY
        im[y_img, x_img] = pixel_values

        return im



class Kitti_Dimension_Reduction():
    """
        Written by Ken_Edward of SEU, on 10/8/2023.

        Function:
        对kitti点云数据进行预处理，使其降维为bev或fv.

        Args:None

        Usage:参见pc_calibration方法与get_calib_dat方法.
    """
    def __init__(self):
        self._class_list = {}
        self._colormap = [matplotlib.cm.get_cmap(name=i) for i in ['Set3']]
        self.max_class = 0

    def generate_bev(self, 
                        pc_data=None,
                        b_boxs=None,
                        box_labels=None,
                        scaling=1,
                        pc_res=0.01,
                        point_size=1,
                        show_bev_mode='img',
                        box_thickness=3,
                        **kargs
                        ):
        """
        Written by Ken_Edward of SEU, on 15/7/2023.

        function:从PC生成BEV

        Args:
            pc_data: 点云数据;数据为2 axis,axis=0为点云中点的数量,
                    axis=1前三通道必须为(x,y,z...),
                    数据类型可为Tensor实例或Ndarray实例.
            b_boxs: 预测后的边界框;数据为2 axis,axis=0为边界框的数量,
                    axis=1通道必须为(x,y,z,w,h,l,theta),
                    数据类型可为Tensor实例或Ndarray实例.
            box_labels: 预测后的边界框对应的标签;元素为str类型,数据为2 axis,
                    axis=0为标签的数量,axis=1通道数据为str类型,
                    数据类型可为Tensor实例或Ndarray实例.
            scaling: BEV的缩放比例,默认为1;
                    但是会卡机,建议大小4以上;
                    数据类型int,可为float.
            pc_res: PC点的分辨率,转化为毫米,float型.
            point_size: 原始PC生成的BEV中点的膨胀倍数;
                    默认为1(1+1),建议为1;若为None,则不进行膨胀;
                    数据类型为int.
            show_bev_mode: BEV显示的方式,数据类型str;
                    默认为'img',显示单张图片并无限等待;若为'video'则会显示视频流.
            box_thickness: 检测边界框宽度,默认为3;数据类型int.

        Returns:
            bev: 二维BEV,结构为标准3 axis BGR图像;数据类型为Ndarray.
        """
        # 数据类型转换为Ndarray
        if isinstance(pc_data, torch.Tensor):
            pc_data = pc_data.cpu().numpy()
        if isinstance(b_boxs, torch.Tensor):
            b_boxs = b_boxs.cpu().numpy()
        if isinstance(box_labels, torch.Tensor):
            box_labels = box_labels.cpu().numpy()
        if pc_data is None:
            raise ValueError("pc_data shouldn't be None.")
        # 生成颜色dict
        if box_labels is not None:
            # print(box_labels)
            for itm in box_labels:
                if itm not in self._class_list:
                    # print(self.max_class//20)
                    color = np.array(self._colormap[self.max_class//10]((self.max_class-self.max_class//10)/10))
                    # print(type(color))
                    # color = tuple([int(i) for i in color[:3]])
                    color = color[:3]
                    self._class_list[itm] = color
                    self.max_class+=1
                    if self.max_class >=10:
                        raise ValueError("The total data classes should be less than 10.")
        
        # x = points[:,0]
        # points[:,0] = points[:,1]
        # points[:,1] = x
        # 生成BEV
        x_min, y_min, z_min = [np.min(pc_data[:,i]) for i in range(3)]
        x_max, y_max, z_max = [np.max(pc_data[:,i]) for i in range(3)]
        for n in range(3):
            pc_data[:,n] = pc_data[:,n]-pc_data[:,n].min()
            pc_data[:, n] = pc_data[:,n]/0.01
        pc_data = np.array(pc_data, dtype=np.int16)
        # print(self.points[:,1].max())
        bev = np.zeros([pc_data[:,0].max()+1, pc_data[:,1].max()+1], dtype=np.int16)
        bev[pc_data[:,0], pc_data[:,1]] = pc_data[:,2]
        # print(np.shape(bev), bev[:,0])
        # plt.imshow(bev)
        # plt.show()
        if point_size is not None:
            kernel = np.ones((3,3),np.uint8)
            bev = cv2.dilate(bev, kernel, iterations = point_size)
        # bev = self._convert_pc_to_bev(point_size=point_size)
        bev_x,bev_y = np.shape(bev)
        img = np.zeros((bev_x,bev_y,3))
        for i in range(3):
            img[:,:,i] = bev
        # 添加检测bboxs
        if b_boxs is not None:
            for n, box in enumerate(b_boxs):
                x,y,_,w,h,_,theta = box
                box = cv2.RotatedRect(((x-x_min)/pc_res,(-y+y_max)/pc_res), 
                                        (w/pc_res,h/pc_res), 
                                        -theta*180/3.1416).points()
                # print(np.shape([box]))
                box = np.array(box,dtype='int32')
                # print(box)
                # print(theta*180/3.1416)
                if box_labels is not None:
                    # print('a')
                    # print(self._class_list[box_labels[n]])
                    # print(type(self._class_list[box_labels[n]][0]))
                    img = cv2.polylines(img, [box], color=self._class_list[box_labels[n]], isClosed=True, thickness=box_thickness)
                else:
                    img = cv2.polylines(img, [box], color=(0,255,0), isClosed=True, thickness=box_thickness)
                # cv2.imshow('bev', cv2.resize(img,(int(bev_y/6),int(bev_x/6))))
                # cv2.waitKey(0)
        img = cv2.resize(img,(int(bev_y/scaling),int(bev_x/scaling)))
        # print(np.shape(img))
        # 显示图片
        if show_bev_mode == 'img':
            cv2.imshow('bev', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif show_bev_mode == 'video':
            cv2.imshow('bev', img)
            cv2.waitKey(1)
        elif show_bev_mode == 'plt':
            plt.imshow(img)
            plt.show()
        return img

    def augment_2D(self, show_img=False):
        """
            function: 进行fv图的数据增强
        """
        kernel = np.ones((3, 5), np.uint8)
        fv = cv2.filter2D(self.mask, -1, kernel)/3
        if show_img:
            plt.imshow(fv, cmap='gray')
            plt.show()

 
def read_velodyne_bin(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)
 
 
def show_point(data):
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(origin)
    # o3d.visualization.draw_geometries([pcd])
 
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])
 
 
def get_one_frame_pc():
    # raw_point = np.loadtxt('./data/modelnet40_normal_resampled/airplane/airplane_0001.txt', delimiter=',').astype(
    #     np.float32)[:, :3]
    # raw_point = np.load('1.npy') #读取1.npy数据  N*[x,y,z]
    raw_point = np.fromfile("/home/ken/workspace/OpenPCDet/tools/000000.bin", dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
    #创建窗口对象
    vis = o3d.visualization.Visualizer()
    #设置窗口标题
    vis.create_window(window_name="kitti")
    #设置点云大小
    vis.get_render_option().point_size = 1
    #设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    #创建点云对象
    pcd=o3d.open3d.geometry.PointCloud()
    #将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    #设置点的颜色为白色
    pcd.paint_uniform_color([1,1,1])
    #将点云加入到窗口中
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
 
def main():
    root_dir = '/home/ken/workspace/OpenPCDet/tools/'  # 激光雷达bin文件路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)
        show_point(db_np)

if __name__ == '__main__':
    pc_dir = '/home/ken/workspace/OpenPCDet/tools/000000.bin'
    # get_one_frame_pc()
    pc = read_velodyne_bin(pc_dir)
    # show_point(db_np)
    a = Draw_BEV()
    bev = a.generate_bev(pc[:,[1,2,0]],
                box_thickness=15,
                scaling = 1,
                show_bev_mode='mpl')
    pass
