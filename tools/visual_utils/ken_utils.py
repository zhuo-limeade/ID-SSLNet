import time, os
from threading import Thread
import numpy as np
import open3d as o3d

class Multi_Thread_Process():
    """
        Written by Ken_Edward of SEU, on 10/8/2023.

        Function:
        对可分割数据类型进行多线程执行操作，仅适用于无return函数.

        Args:
            func: 需要执行的函数
            do_list: 参数list
            thread_num: 需要占用的CPU线程数

        Usage:
        先执行Multi_Thread_Process.init_thread方法,再执行先执行Multi_Thread_Process.run方法
    """
    def __init__(self,  func, do_list, thread_num=1):

        self.func = func
        self.thread_dict = {}
        self.do_list = do_list
        self.do_list_len = len(self.do_list)
        self.thread_num = thread_num

    def init_thread(self):
        for n in range(self.thread_num):
            self.thread_dict['t' + str(n)] = Thread(
                target=self.func,
                args=(self.do_list[self.do_list_len * n // self.thread_num:self.do_list_len * (n + 1) // self.thread_num], ))

    def run(self):
        for thrd in self.thread_dict:
            self.thread_dict[thrd].start()

def check_dir_existence(dir):
    """
        Written by Ken_Edward of SEU, on 10/8/2023.

        Function:
        检查目录是否存在,不存在会自动生成一个.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def show_pc(pc):
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
    pcd.points= o3d.open3d.utility.Vector3dVector(pc)
    #设置点的颜色为白色
    pcd.paint_uniform_color([1,1,1])
    #将点云加入到窗口中
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    def test_func(a, **krags):
        print('a')

    mtp_test = Multi_Thread_Process(test_func, do_list=[1,2,3], thread_num=3)
    mtp_test.init_thread()
    mtp_test.run()

