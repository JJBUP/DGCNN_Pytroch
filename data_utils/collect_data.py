import os
import sys
from data_util import  collect_point_label

# 1. 路径
DATA_PATH = "/data/Stanford3dDataset_v1.2_Aligned_Version_fixed"
# BASE_DIR 为 collect_indoor3d_data.py的文件夹的上一层data_utils 绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(
    __file__))  # BASE_DIR: 'E:\\PythonProjects\\PointCloud\\classification\\Pointnet2_pytorch\\data_utils'
# ROOT_DIR 为BASE_DIR 上一层文件夹的绝对路径
ROOT_DIR = os.path.dirname(BASE_DIR)  # ROOT_DIR:'E:\\PythonProjects\\PointCloud\\classification\\Pointnet2_pytorch'

sys.path.append(BASE_DIR)

# 2.获取 meta 文件夹中anno_paths.txt保存了原始S3DIS数据注释摆放格式
anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
# 拼接数据集data_path 和 Annotations 位置
# DATA_PATH:'E:\\PythonProjects\\PointCloud\\classification\\Pointnet2_pytorch\\data\\s3dis\\Stanford3dDataset_v1.2_Aligned_Version'
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

# 3.创建修整S3DIS后的文件夹路径 root + data/stanford_indoor3d
output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
# 注：v1.2数据中有一个额外字符，位于Area_5/hallway_6。它是手动修复的。
for anno_path in anno_paths:
    print("正在处理 ：" + anno_path)  # 每一个注释的路径

    # 注意：linux和windows下分隔符不同,这里做了优化，具体值可以debug
    # windows:
    # elements = anno_path.split('\\')
    # elements = elements[-1].split('/')
    # linux:
    elements = anno_path.split('/')

    # 设置输出文件名
    # out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # Area_1_hallway_1.npy
    out_filename = elements[-3] + '_' + elements[-2] + '.txt'  # Area_1_hallway_1.txt
    collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'txt')
print("处理完毕")