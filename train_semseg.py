"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils.SceneDataset import RandomAllSceneDataset
import torch
from datetime import datetime
import logging  # 加载日志包
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

# 获得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


# sys.path是一个列表list，它里面包含了已经添加到系统的环境变量路径
# 当我们要添加自己的引用模块搜索目录时，可以通过list的append方法
# 作用：importlib.import_module()

def parse_args():
    parser = argparse.ArgumentParser(description="train model")
    # 数据集类别，序号即id号，注意：其应该与data_utils/meta/calssnames.txt的语义分割标签名称一致，数量一致，
    # 若想只分割一部分，应该在dataset中将不在列表中的类归为杂项
    parser.add_argument('--class_list', type=list,
                        default=['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair',
                                 'sofa', 'bookcase',
                                 'board', 'clutter'],
                        help='class_list')
    # 设置语义分割模型，本代码中可以设置pointnet_sem_seg_ssg和pointnet_sem_seg_msg
    parser.add_argument('--model', type=str, default='DGCNN_semseg',
                        help='model name [default: pointnet_sem_seg]')
    # batch_size大小
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
    # knn
    parser.add_argument('--k', type=int, default=20, help='KNN [default: 20]')
    # epoch
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 32]')
    # 线程数number_work
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader work number')
    # 是否将数据驻留在内存中
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    # 单GPU
    parser.add_argument('--cuda', type=str, default='1', help='GPU to use [default: GPU 0]')
    # S3DIS(indoor3d) 数据集路径
    parser.add_argument('--dataset_root', type=str,
                        default='./data/stanford_indoor3d_test',
                        help='the dataset root dir of S3DIS(indoor3d)')
    # xyz坐标的缩放比例用于提高单位blocks中点的数目
    parser.add_argument('--scale', type=tuple, default=(1, 1, 1), help='scale dataset for xyz')
    # 数据集中验证数据集(相反剩下的为训练)
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    # 中断训练的文件夹,None将从头开始训练，若设置为log/sem_seg/下的目录名称如：2022-07-14_14-16-10-msg则可以继续训练
    parser.add_argument('--resume_log_dir', type=str, default=None, help='Log path [default: None]')
    # learning_rate
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD [default: Adam]')
    # 权重衰减
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    # 读取一个点云点的数目
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    # 学习率衰减步长(单位epoch)
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    # 权重衰减，还没看怎么用
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()


# relu 的inplace参数，true表示直接在原始数据上将小于0的部分设置为0，反之复制一份新数据


def main(args):
    def log_string(str):
        # 用于looging 保存str和控制台输出str
        logging.info(str)  # 记录
        print(str)  # 控制台输出

    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace = True

    # 默认权重初始化函数，权重默认使用HE初始化这里使用xavier
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            # torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            # torch.nn.init.constant_(m.bias.data, 0.0)

    # bn层均值方差的衰减，因为早期训练未收敛bn层均值方差价值较小
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    '''HYPER PARAMETER'''
    # 设置使用的gpu给环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    '''CREATE DIR'''
    # dir tree
    # ./log
    #  |-classification
    #  |-part_seg
    #  |-sem-seg
    #       |-2020-7-06 19:18:15
    #       |-checkpoint
    #       |      |-best_model.pth
    #       |-logs
    #          |-eval.txt
    #          |-网络model:pointnet2_sem_seg.py
    #          |-网络utils:pointnet2_utils.py

    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)

    # 是否使用原log文件夹地址（记录log和设置中断训练）
    if args.resume_log_dir is None:
        timestr = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.resume_log_dir)
    experiment_dir.mkdir(exist_ok=True)
    # 中断训练检查点的文件路径
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    # log文件路径
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    # tensorboard 文件路径
    tensorboard_dir = experiment_dir.joinpath('tensorboard/')
    tensorboard_dir.mkdir(exist_ok=True)

    '''LOG'''
    # （学习）设置log记录，用于记录valid时的数据
    # 设置日志输出级别、路径、格式
    logging.basicConfig(level=logging.INFO,
                        filename='%s/train_log.txt' % log_dir,
                        format='%(asctime)s %(message)s')
    log_string('PARAMETER ...')
    args = parse_args()
    log_string(args)
    # 设置tensorboard
    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # 设置id与cls的转换列表，用于记录到log和tensorboard中
    # label(id)转class(name),根据id(label)取名称(class)，{cell:0,floor:1 ... }
    seg_label_to_class = {i: cls for i, cls in enumerate(args.class_list)}

    NUM_CLASSES = len(args.class_list)  # indoor3D is 13
    NUM_POINT = args.npoint  # 单个数据采样点的数量
    BATCH_SIZE = args.batch_size  # Batch size大小

    # 读取点云，该中方式init时将点云全部读入到内存，可以init只记录路径，等getitem时再去获取将会减轻内存压力
    print("------start loading training data ...")
    TRAIN_DATASET = RandomAllSceneDataset(is_train=True, data_root=args.dataset_root, block_points_num=NUM_POINT,
                                          scale=args.scale, test_area=args.test_area, class_num=len(args.class_list),
                                          block_size=1.0, sample_rate=1.0, transform=None)
    # 读取点云，该中方式init时将点云全部读入到内存，可以init只记录路径，等getitem时再去获取将会减轻内存压力
    print("------start loading test data ...")
    TEST_DATASET = RandomAllSceneDataset(is_train=False, data_root=args.dataset_root, block_points_num=NUM_POINT,
                                         scale=args.scale, test_area=args.test_area, class_num=len(args.class_list),
                                         block_size=1.0, sample_rate=1.0, transform=None)
    # 训练数据加载器
    train_dataLoader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=args.pin_memory, drop_last=False,
                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    # 测试数据加载器
    test_dataLoader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=args.pin_memory, drop_last=False)
    # 获取label标签的权重，在loss反向传播时使用
    label_weights = torch.Tensor(TRAIN_DATASET.labelweights).to(device)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    # 复制当前使用的model 和model_utils到log 中

    copy_model_path = experiment_dir.joinpath('model/')
    copy_model_path.mkdir(exist_ok=True)
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(copy_model_path))
    shutil.copy('./models/DGCNN_utils.py', str(copy_model_path))


    # 获得模型，注意该模型获取方式是动态的
    classifier = MODEL.get_model(k=20,num_cls=13).to(device)
    # 获得loss计算的模型
    criterion = MODEL.get_loss().to(device)
    # apply:递归的将函数fn应用到model 的每一个子模块(创建模型的时候不调整，这里纯粹是在炫技是吧)
    classifier.apply(inplace_relu)

    # （学习）加载预训练模型
    try:
        # 尝试加载预训练模型位置 ./log/checkpoints/best_model.pth
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch'] + 1  # epoch
        classifier.load_state_dict(checkpoint['model_state_dict'])  # 模型参数
        # 注意：优化器的学习率衰减由epoch 计算，且优化器参数没有发生变化故没有从模型中加载
        # 若学习率由pytoch提供的lr_shedule控制则需要从模型中加载
        log_string('Use pretrain model')
    except:
        # 加载异常则显示不存在model
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
        # 注意：优于优化器的学习率和
    # 优化器选择
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9,
                                    weight_decay=args.decay_rate)

    LEARNING_RATE_CLIP = 1e-5  # 学习率剪子，即最小学习率
    MOMENTUM_ORIGINAL = 0.1  # bn层原始衰减动量
    MOMENTUM_DECCAY = 0.5  # bn层权重衰减动量
    MOMENTUM_DECCAY_STEP = args.step_size  # bn层权重衰减步长

    global_epoch = 0  #
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes 训练切碎的点云场景'''
        # 训练开始
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        # 计算学习率
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        # 记录并输出学习率
        log_string('Learning rate:%f' % lr)
        # 找到优化器中学习率参数，注意优化其中保存着我们待更新的参数和以及学习率权重衰减等一系列参数
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:  # bn层权重衰减动量最小值
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        # (学习)设置BN层均值和方差衰减(第一次见到这样设置，很不错！)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        total_correct = 0  # one epoch 预测正确的点数目
        total_seen = 0  # one epoch 所训练(seen看到)的总点的数目
        loss_sum = 0  # one epoch loss的总和

        classifier = classifier.train()
        torch.no_grad()
        torch.autograd.no_grad()
        # 开始当前epoch 训练(for --> 取batch数据)
        for i, (points, target) in tqdm(iterable=enumerate(train_dataLoader), total=len(train_dataLoader),
                                        colour="BLUE", desc="epoch[{}/{}]-train".format(start_epoch + 1, args.epoch)):
            # points, target 均为numpy格式
            optimizer.zero_grad()

            points = points.data.numpy()
            # 数据增强随机旋转点云的z轴
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            # print(points.dim())
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            # 模型预测，seg_pred为每个点的预测[B,N,num_class](调整了通道),trans_feat为最后一层SA层输出特征[B,D.S]
            seg_pred = classifier(points)
            # 调整pred 和targe 结果形状，用于计算loss，因为loss只能计算2d的
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)  # [B,N,CLASS]-->[B*N,CLASS]
            target = target.view(-1)  # [B,N]-->[B*N]
            # 计算loss
            loss = criterion(seg_pred, target, label_weights)
            # 反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 累计loss
            loss_sum += loss

            # 提取每个点预测的最大值作为预测结果 [B*N,CLASS] -->[B*N]
            pred_choice = seg_pred.cpu().data.argmax(1).numpy()
            target_np = target.cpu().data.numpy()
            # 计算该batch 的正确率
            correct = np.sum(pred_choice == target_np)
            # 累计 的预测对的点正确个数
            total_correct += correct
            # 累计 所见点云点的总数累计
            total_seen += (BATCH_SIZE * NUM_POINT)

        # ------计算train的loss、accuracy------
        # 计算 总体精度OA 平均类精度mAcc
        # OA ,overall accuracy，总体准确率，[epoch中正确的点/总点]
        OA = total_correct / float(total_seen)
        # mLoss ,mean loss ,平均损失
        mLoss = loss_sum / float(len(train_dataLoader))
        log_string('Training mean loss: {}'.format(mLoss))
        log_string('Training accuracy: {}'.format(OA))
        tags = ["train/loss", "train/overall accuracy"]
        for data, tag in zip([mLoss, OA], tags):
            tb_writer.add_scalar(tag=tag, scalar_value=data, global_step=epoch)

        if epoch % 5 == 0:
            logging.info('Save model...')
            # 保存模型参数和分类状态参数的路径
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            # (学习)保存模型当前状态，可以间断学习
            state = {
                # 本训练流程的学习率/bn层权重衰减是通过epoch生成的
                # 我们可以使用pytorch提供的lr_shedule学习率调度器来优化学习率并用state_dict( )来保存
                'epoch': epoch,
                # 保存优化器
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': classifier.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        # 验证
        '''Evaluate on chopped scenes'''
        with torch.no_grad():  # 强制之后的内容不进行计算图构建

            #
            num_batches = len(test_dataLoader)
            # epoch 总点统计
            total_correct = 0  # one epoch 预测正确的点数目
            total_seen = 0  # one epoch 所训练(seen看到)的总点的数目
            loss_sum = 0  # one epoch loss的总和
            # epoch 级别class分类统计
            labelweights = np.zeros(NUM_CLASSES)  # 标签权重，暂时初始化为[0,0,0,0...]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]  # total_correct_class 表示所有正确的点格局class分类的数量
            total_seen_class = [0 for _ in range(NUM_CLASSES)]  # total_seen_class表示所有训练的点根据class分类的数量
            total_union_class = [0 for _ in range(NUM_CLASSES)]  # total_union_class 表示所有预测correct和标签correct并集，按照class分类

            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(test_dataLoader), total=len(test_dataLoader), colour="BLUE",
                                            desc="epoch[{}/{}]-train".format(start_epoch + 1, args.epoch)):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().to(device), target.long().to(device)
                points = points.transpose(2, 1)  # 转换1,2维度 [B,N,D]-->[B,D,N]
                # 模型预测
                seg_pred, trans_feat = classifier(points)
                # 改变输出形式用于后续计算
                seg_pred_np = seg_pred.contiguous().cpu().data.numpy()  # [B,N,CLASS]
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)  # [B*N,CLASS]
                target_np = target.cpu().data.numpy()  # [B,N,CLASS]
                target = target.view(-1, 1)[:, 0]  # [B*N,CLASS]

                # ------计算loss、accuracy、iou------
                # 计算loss
                loss = criterion(seg_pred, target, trans_feat, label_weights)
                loss_sum += loss
                # 计算correct
                seg_pred_np = np.argmax(seg_pred_np, 2)  # [B,N,CLASS]-->[B,N] 返回的是索引
                correct = np.sum((seg_pred_np == target_np))
                total_correct += correct
                # 计算epoch总共处理点数目
                total_seen += (BATCH_SIZE * NUM_POINT)
                # 统计 该epoch中 点的各标签数量 [label1_size,label2_size]
                # 注意：不是整个train或valid数据集的label权重
                tmp, _ = np.histogram(target_np, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((target_np == l))  # 预测类别数量[cls1_num,cls2num....]
                    # 预测正确数量[cls1_num,cls2num....]
                    # &表示按位与，True 表示1，False表示0，这里的|&和||&&作用是一样的，我可以人为他又在炫技吗？
                    total_correct_class[l] += np.sum((seg_pred_np == l) & (target_np == l))
                    # 预测为l 和 标签为l 的总数量[cls1_num,cls2num....]  miou的并集
                    # |表示按位或
                    total_union_class[l] += np.sum(((seg_pred_np == l) | (target_np == l)))
            # 计算标签权重,
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))

            # 计算每个分类的IoU 类名+epoch中点的标签权重+类别IoU
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                class_name = seg_label_to_class[l]  # 类别名
                class_IoU = total_correct_class[l] / (float(total_union_class[l] + 1e-6))  # 类别IoU
                indent = ' ' * (14 - len(seg_label_to_class[l]))  # 缩进
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    # code: ' ' * (14 - len(seg_label_to_class[l]))控制缩进
                    class_name + indent, labelweights[l - 1], class_IoU
                )

                tb_writer.add_scalar(tag='valid/' + class_name + " IoU", scalar_value=class_IoU, global_step=epoch)

            log_string(iou_per_class_str)

            # 计算 总体精度OA 平均类精度mAcc
            # OA ,overall accuracy，总体准确率，[epoch中正确的点/总点]
            OA = total_correct / float(total_seen)
            # mAcc ,mean class accuracy ,平均类别准确率，[求每个类的准确率(该类正确的点/总点)，再按类别求平均]
            mAcc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))
            # mIoU ,平均类别的IoU,[ 按类别求 预测正确的点(交)/预测为l的点+标签为l的点的并(并) ]
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_union_class, dtype=np.float64) + 1e-6))
            # mLoss ,mean loss ,平均损失
            mLoss = loss_sum / float(num_batches)

            log_string('eval mean loss: {}'.format(mLoss))  # mLoss
            log_string('eval point accuracy: {}'.format(OA))  # OA
            log_string('eval point avg class acc: {}'.format(mAcc))  # mAcc
            log_string('eval point avg class IoU: {}'.format(mIoU))  # mIoU

            tags = ["valid/loss", "valid/overall accuracy", "valid/mean class accuracy", "valid/mean class IoU"]
            for data, tag in zip([mLoss, OA, mAcc, mIoU], tags):
                tb_writer.add_scalar(tag=tag, scalar_value=data, global_step=epoch)

            # 保存最佳IoU下的模型并记录log
            if mIoU >= best_iou:
                best_iou = mIoU
                logging.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    # 参数
    args = parse_args()
    main(args)
