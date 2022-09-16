"""
jjb
"""
import argparse
import os
from data_utils.SceneDataset import ScanOneSceneDataset
from data_utils.data_util import g_label2color
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


# sys.path是一个列表list，它里面包含了已经添加到系统的环境变量路径
# 当我们要添加自己的引用模块搜索目录时，可以通过list的append方法
# 作用：importlib.import_module()


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    # model 类型，优化器类型根据训练数据自动获取
    # 数据集类别，序号即id号，注意：其应该与data_utils/meta/calssnames.txt的语义分割标签名称一致，数量一致，
    # 若想只分割一部分，应该在dataset中将不在列表中的类归为杂项
    parser.add_argument('--class_list', type=list,
                        default=['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair',
                                 'sofa', 'bookcase',
                                 'board', 'clutter'],
                        help='class_list')
    # 训练记录数据的dir，具体路径怎么设置建议看一下代码
    parser.add_argument('--log_dir', type=str, default="2022-08-24_14-10-33-msg-dp",
                        help='experiment root,如:[2022-07-14_17-05-40]')
    # S3DIS(indoor3d) 数据集路径
    parser.add_argument('--dataset_root', type=str,
                        default='./data/stanford_indoor3d_test',
                        help='the dataset root dir of S3DIS(indoor3d)')
    # xyz坐标的缩放比例用于提高单位blocks中点的数目
    parser.add_argument('--scale', type=tuple, default=(1, 1, 1), help='scale dataset for xyz')
    # 测试数据集
    parser.add_argument('--test_area', type=int, default=1, help='area for testing, option: 1-6 [default: 5]')
    # batchsize
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in testing [default: 32]')
    # 使用gpu
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # 读取一个点云点的数目
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    # 线程数number_work
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader work number')
    # 是否将数据驻留在内存中
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    # 循环投票方式测试次数
    parser.add_argument('--num_votes', type=int, default=1,
                        help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label):
    """
    vote_label_pool:投票的统计池，shape(points,num_class)
    point_idx:点云的索引，即pred_laebl相同位置真正的索引编号，因为经过前期数据补充后发生了改变
    """
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER 超参数'''
    NUM_CLASSES = len(args.class_list)  # indoor3D is 13
    # 设置系统可见gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 训练数据位置
    experiment_dir = 'log/sem_seg/' + args.log_dir
    # 创建可视化文件夹
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)
    # 创建logs文件夹
    logs_dir = experiment_dir + '/logs/'
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(exist_ok=True)

    '''LOG 记录'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler('%s/logs/test_log.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # label(id)转class(name),根据id(label)取名称(class)，{cell:0,floor:1 ... }
    seg_label_to_class = {i: cls for i, cls in enumerate(args.class_list)}

    '''MODEL LOADING 模型加载'''
    # 获得模型名称，log 中model文件夹中查找model名
    if os.listdir(experiment_dir + '/model')[0].split('.')[0].find('utils') == -1:
        model_name = os.listdir(experiment_dir + '/model')[0].split('.')[0]
    else:
        model_name = os.listdir(experiment_dir + '/model')[1].split('.')[0]

    # 动态道路模型py文件
    MODEL = importlib.import_module(model_name)
    # 获得模型
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    # 加载检查点
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # 加载模型参数
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        # 表现代码通用，我们将room 写作 scene
        # ---测试数据集---
        # 扫描dataste根路径，收集独立scene下的点云
        log_string('---- EVALUATION SCENE----')
        scene_file_name_list = os.listdir(path=args.dataset_root)
        log_string("The number of test data is: %d" % len(scene_file_name_list))
        scene_file_name_list = [file_name for file_name in scene_file_name_list
                                if file_name.find("Area_{}".format(args.test_area)) != -1]

        # 统计 所有场景下的iou acc等
        total_seen_class = [0] * NUM_CLASSES  # num_batches中所有点的数目
        total_correct_class = [0] * NUM_CLASSES  # ~ 所有正确点的数目
        total_union_class = [0] * NUM_CLASSES  # ~ 预测为l的点和标签为l的点的并集
        # 处理n个场景数据
        for scene_idx, scene_file_name in enumerate(scene_file_name_list):
            scene_file_path = os.path.join(args.dataset_root, scene_file_name)
            log_string("Inference room: [{}/{}], now is {} ".format(scene_idx + 1,
                                                                    len(scene_file_name_list),
                                                                    scene_file_name))
            # 统计 本场景下的iou acc等
            total_seen_class_one_scene = [0] * NUM_CLASSES  # scene中可见类的数目
            total_correct_class_one_scene = [0] * NUM_CLASSES  # scene中预测正确的点数目
            total_union_class_one_scene = [0] * NUM_CLASSES  # scene中预测和真实的并集
            # 原始场景数据
            labels_scene = None
            points_scene = None
            coord_min = None
            # 真实点云投票池
            vote_label_pool = None
            # 可视化文件写入工具
            fout = open(os.path.join(visual_dir, scene_file_name + '_pred.txt'), 'w')
            fout_gt = open(os.path.join(visual_dir, scene_file_name + '_gt.txt'), 'w')

            # 投票循环预测场景
            for v in range(args.num_votes):
                log_string(" VOTE: [{}/{}]".format(v + 1, args.num_votes))
                # [] 调用getitem 获得dataset数据
                scanOneSceneDataset = ScanOneSceneDataset(data_path=scene_file_path, scale=args.scale,
                                                          block_points_num=args.num_point)
                dataLoader = DataLoader(dataset=scanOneSceneDataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=args.pin_memory)
                # 原始场景数据
                labels_scene = scanOneSceneDataset.semantic_labels  # 场景原始点云标签 [N]
                points_scene = scanOneSceneDataset.points  # 场景原始点云 xyzrgb [N,6]
                coord_min = scanOneSceneDataset.coord_min  # 原始房间数据平移
                points_num = scanOneSceneDataset.points_num  # 场景点云数目
                coord_scale = scanOneSceneDataset.scale  # 场景缩放
                # 投票的方式选择点的真正label [N_scene,cls]
                vote_label_pool = np.zeros((points_num, NUM_CLASSES))

                for points_block_batch, label_block_batch, index_block_batch in tqdm(dataLoader,
                                                                                     desc="PREDICATE",
                                                                                     colour="BLUE"):
                    # label_block_batch   [B,N]
                    # index_block_batch   [B,N]
                    # points_block_batch  [B,N:4096,D--->B,D,N]
                    data = torch.Tensor(points_block_batch).float().transpose(2, 1).cuda()
                    seg_pred = classifier(data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    # 投票的方式选择点的真正标签
                    # 因为dataset每次初始化，会重新调用init，从而产生随机的block
                    # 产生多个点的情况如下：
                    # 1.参数：循环投票次数(主要)
                    # 2.步长：1*1的场景以0.5的步长移动
                    # 3.扩展：移动时的padding
                    # 4.填充：不足num_points(4096)的点会进行填充，(dataset line:268)

                    # 注意:下面中通过index_batch，我们已经将dataset随机/复制过的点 映射 到真实场景scene的idx
                    # vote_label_pool: [N_scene, cls]
                    vote_label_pool = add_vote(vote_label_pool, index_block_batch, batch_pred_label)

            # pred_label:选择投票最多的label作为 该点的真实label
            pred_label_scene = np.argmax(vote_label_pool, 1)
            # 统计 one scene
            for l in range(NUM_CLASSES):
                # 按类别统计 one scene
                total_seen_class_one_scene[l] += np.sum((labels_scene == l))  # 该类别点云点的数目[l]
                total_correct_class_one_scene[l] += np.sum((pred_label_scene == l) & (labels_scene == l))  # 该类别并数目[l]
                total_union_class_one_scene[l] += np.sum(((pred_label_scene == l) | (labels_scene == l)))  # 该类别交数目[l]
                # 按类别统计 all scene(累计)
                total_seen_class[l] += total_seen_class_one_scene[l]
                total_correct_class[l] += total_correct_class_one_scene[l]
                total_union_class[l] += total_union_class_one_scene[l]
            log_string("-----one scene result-----")
            # 单个场景下 所有类别iou iou_class_one_scene [l]
            iou_class_one_scene = np.array(total_correct_class_one_scene) / (
                    np.array(total_union_class_one_scene, dtype=np.float) + 1e-6)
            log_string("classes IoU  of {} :\n{}".format(scene_file_name, iou_class_one_scene))

            # 单个场景下 平均类别iou iou_mean_class_one_scene [1]
            valid = np.array(total_seen_class_one_scene)  # 保证点云中有该类别的点才进行mean iou
            iou_mean_class_one_scene = np.mean(iou_class_one_scene[valid != 0])
            log_string('Mean classes IoU of {}: {}'.format(scene_file_name, iou_mean_class_one_scene))

            # 按 点顺序 保存 预测标签
            filename = os.path.join(visual_dir, scene_file_name + "_pred_id" + '.txt')
            with open(filename, 'w') as pred_label_save:
                for i in pred_label_scene:
                    pred_label_save.write(str(int(i)) + '\n')
                pred_label_save.close()
            # 处理每个点的数据
            for i in range(points_scene.shape[0]):
                # 获得预测标签对应的颜色
                color = g_label2color[pred_label_scene[i]]
                # 获得真实标签对应的颜色
                color_gt = g_label2color[labels_scene[i]]
                # 保存修改过颜色后的点云
                fout.write('%f %f %f %d %d %d\n' % (
                    points_scene[i, 0] / coord_scale[0] + coord_min[0],  # (偏移后坐标/缩放)+偏移量
                    points_scene[i, 1] / coord_scale[1] + coord_min[1],
                    points_scene[i, 2] / coord_scale[2] + coord_min[2],
                    color[0], color[1], color[2]))
                fout_gt.write('%f %f %f %d %d %d\n' % (
                    points_scene[i, 0] / coord_scale[0] + coord_min[0],  # (偏移后坐标/缩放)+偏移量
                    points_scene[i, 1] / coord_scale[1] + coord_min[1],
                    points_scene[i, 2] / coord_scale[2] + coord_min[2],
                    color_gt[0], color_gt[1], color_gt[2]))
            fout.close()
            fout_gt.close()
        # for all scene end

        log_string("-----all scene result-----")
        # 统计 all scene
        # 所有场景下 所有类别iou
        for l in range(NUM_CLASSES):
            log_string('class {}, IoU: {}'.format(
                seg_label_to_class[l] + ' ' * (14 - len(seg_label_to_class[l])),
                total_correct_class[l] / float(total_union_class[l])))
        iou_class_all_scene = np.array(total_correct_class) / (np.array(total_union_class, dtype=np.float64) + 1e-6)
        # 平均 IoU,平均类别accuracy ，平均 accuracy
        mIoU = np.mean(iou_class_all_scene)
        mAcc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))
        OA = np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)
        log_string('mean class IoU: {}'.format(mIoU))
        log_string('avg class acc: {}'.format(mAcc))
        log_string('overall accuracy: {}'.format(OA))

        log_string("-----Finished-----")


if __name__ == '__main__':
    args = parse_args()
    main(args)
