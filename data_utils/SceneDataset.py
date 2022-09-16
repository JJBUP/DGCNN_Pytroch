import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


# S3DISDataset 的数据集
# 该数据集是用于室内分割的数据集
# 使用该数据集前需要先进行整理，将其分割为1m*1m的区域进行分割
# 修改方法: project/data_utils/collect_indoor3d_data.py
# 注意：由于S3DIS数据集文件夹名称变为indoor3d，下面数据集为了区分变之前和之后，我们使用为indoor3d作为数据集名称

# 用于train，读取文件夹下所有scene作为1个dataset，根据权重选择scene+随机挑选scene的点训练
class RandomAllSceneDataset(Dataset):
    def __init__(self, is_train=True, data_root='train_val_root数据集根目录', class_num=13, block_points_num=4096,
                 scale=(1, 1, 1), test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        """
        is_train : 分开，表示是区分train  test,默认为True
        data_root: 数据集根目录
        class_num: 训练类别数
        block_points_num: 从房间中采样1m*1m点云（block）规定提取点的数目，(如果block>4096则继续采样到4096，block<4096则重复采样至4096)
        test_area: 选定第几个Area的房间作为训练数据，默认为第5个，用于区分which train dataset，which test dataset
        block_size: 在房间中截取区域的大小默认为 x y 为 1m*1m，z轴不做切分
        sample_rate: 采样率，采样率对整体点云的点进行采样，计算block的个数，采样率越大block越少多，则一个epoch中训练数据也就越多
        scale=(0.01, 0.01, 0.01): 点云数据缩放比例，若点云数据范围或点间隔较大，通过缩放点云xyz大小比例实现数据缩小，更小的数据值训练越快
        提示：加大采样率是不是能够更充分的利用数据？可以试一下

        transform: 传递对象，该对象能够实现点云的变换(代码中未使用)

        """

        super().__init__()
        self.block_points_num = block_points_num
        self.block_size = block_size
        self.transform = transform
        self.scale = scale
        # 1.读取文件名
        # 对indoor3d 数据集内‘文件名称’进行排序，防止不同系统顺序不同
        scene_filename_list = sorted(os.listdir(data_root))
        # 修整rooms中记录的文件，如果文件名不包括area_则认为杂项文件将其剔除
        scene_filename_list = [f for f in scene_filename_list if 'Area_' in f]

        # 2.划分数据集 默认area_5区域的为测试数据集
        if is_train is True:
            scene_filename_list = [f for f in scene_filename_list if not 'Area_{}'.format(test_area) in f]
        else:
            scene_filename_list = [f for f in scene_filename_list if 'Area_{}'.format(test_area) in f]
        self.scene_points_list, self.scene_labels_list = [], []  # 保存点云数据和标签数据
        self.scene_coord_min_list = []  # 保存点小坐标(3),用作平移数据
        self.scene_points_num_list = []  # 统计每个场景下点的数目
        labelweights = np.zeros(class_num)

        # 3. 读取点云文件 points label
        # 读取room_split[]中的文件名
        for scene_filename in tqdm(scene_filename_list, total=len(scene_filename_list), desc="data conversion",
                                   colour="GREEN"):
            # 拼接点云数据的完整路径 保存为room_path列表
            scene_path = os.path.join(data_root, scene_filename)
            # 加载点云数据np
            scene_data = np.loadtxt(scene_path)  # xyzrgbl, N*7
            # 去0-5作为点云数据，6作为label
            points, labels = scene_data[:, 0:6], scene_data[:, 6]  # xyzrgb, N*6; l, N
            # 数据偏移，使用最小值
            coord_min = np.amin(points[:, :3], axis=0)
            points[:, :3] = points[:, :3] - coord_min[None, :]  # 偏移 [N,3] - [1,3]
            # 数据缩放，使用超参数scale
            points[:, :3] = points[:, :3] * self.scale
            # label：根据label列创建直方图
            # 直方图x轴: 1 l1 2 l2 3 l3 4 l4 5 l5 6 l6 7 l7 8 l8 9 l9 10 l10 11 l11 12 l12 13
            hist, _ = np.histogram(a=labels, bins=range(class_num + 1))  # a表示带统计列表，bin 表示区间0-13 共有13个区间
            # 统计 每个area区域的label个数作为label权重
            labelweights += hist
            # 保存各类数据
            self.scene_coord_min_list.append(coord_min)  # [area1[min_x,min_y,min_z],area2[...]...]
            self.scene_points_list.append(points)  # [area1[ [xyzrgb]  , [xyzrgb] ...], area2[...]...]
            self.scene_labels_list.append(labels)  # [area1[    l1,         l2    ...], area2[...]...]
            self.scene_points_num_list.append(labels.size)  # [ area1_a_size ,area1_b_size....]

        # 此时读取完所有点云文件

        # 4. 计算标签权重
        # labelweights[ area1[l1,l2,l3...],area2[l1,l2,l3...] ..]
        labelweights = labelweights.astype(np.float32) + 1e-6  # 防止除数为0
        # 将统计的label数量归一化转化为权重
        labelweights = labelweights / np.sum(labelweights)
        # (label_max / [label1, label2 ...] )^(1/3) ,^(1/3)我认为应该是防止权重过大吧
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        # 展示一下权重
        # class_names = np.loadtxt("./data_utils/meta/class_names.txt", dtype=str)
        class_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair',
                       'sofa', 'bookcase', 'board', 'clutter']
        name_weigth = {names: round(weigth, 3) for names, weigth in zip(class_names, self.labelweights.tolist())}
        print("label weight: ", name_weigth)

        # 5，生成房间索引 scene_idx

        # 5.1 计算每个scene的点占all scene点的比例scene_num_rate,作为房间权重
        scene_num_rate = self.scene_points_num_list / np.sum(self.scene_points_num_list)
        # 5.2 计算点云训练所用block数目
        # scene_points_num_list: 读入所有房间scene的点云的数目 [ area1_room1_size ,area2_room1_size,....]
        # sample_rate:表示采样比例，该参数能够缩小block_num_iter数量
        # block_points_num: 一次采样4096个点
        # block_num_iter: 整个数据集有多少次4096个点的数目，可以代表为点云数量(因为我们一次采样4096个点，采样方法见getitem)
        block_num_iter = int(np.sum(self.scene_points_num_list) * sample_rate / self.block_points_num)

        # 5.3 计算在block_num_iter数目下，按照比例scene_num_rate，idx应该选取那个房间的4096个点
        # scene_idxs: 用于dataloader 采样 [训练用的点云(4096点)]  使用的是 [which scene 中的点] 的映射
        #            解释: getitem(idx), idx表示取第idx个点云
        #                 idx 作为 scene_idxs的索引 scene_idxs[idx]
        #                 scene_idxs:[0,0,1,1,1,2,3,3],scene_idxs[]中数值为数据集中scene的索引scene_idx
        #                 所以，idxs + scene_idxs 可以获得我们在那个scene点云中采样一次点云数据(4096点)用作训练
        scene_idxs = []
        for index in range(len(scene_filename_list)):  # scene_filename_list 为训练或验证下scene的数目 ，index为当前scene的编号
            # scene_num_rate[index] * block_num_iter 房间点比例 * 总blocks数目 = 每个房间的block数目
            # 列表乘法 [1]* 5 =[1,1,1,1,1]
            scene_idxs.extend([index] * int(round(scene_num_rate[index] * block_num_iter)))
        self.scene_idxs = np.array(scene_idxs)
        print("Totally {} samples in {} set.".format(len(self.scene_idxs), "Train" if is_train else "Test"))

    def __getitem__(self, idx):
        scene_idx = self.scene_idxs[idx]  # 用于dataloader 采样 [训练用的点云(4096点)]  使用的是 [which scene 中的点] 的映射
        scene_points = self.scene_points_list[scene_idx]  # points: [N , 6]
        scene_labels = self.scene_labels_list[scene_idx]  # labels: [N]
        scene_point_num = scene_points.shape[0]  # N 该scene点云点的数目
        scene_coord_max = np.amax(scene_points, axis=0)  # init 平移后的 room max

        # 在 scene_idx 指定房间中采样block(默认4096个点)
        while (True):
            # 随机选取1m*1m的点作为样本的中心点[xyz]
            center = scene_points[np.random.choice(scene_point_num)][:3]
            # 计算1m*1m距离的最大值[xyz]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            # 计算1m*1m距离的最小值[xyz]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            # 获得1m*1m区域点的坐标(z轴不受限制)
            blocks_points_idxs = np.where(
                (scene_points[:, 0] >= block_min[0]) & (scene_points[:, 0] <= block_max[0]) & (
                        scene_points[:, 1] >= block_min[1]) & (
                        scene_points[:, 1] <= block_max[1]))[0]
            # 如果1m*1m区域内point_idxs 点的数量大于1024个点就跳出循环，1024是超参数
            if blocks_points_idxs.size > 1024:
                # 目的：保证中心点1m*1m范围内至少有1024个点
                break

        # 在选定的block中采样
        # 目的：由于输入点的数量必须一致，所以扩增
        if blocks_points_idxs.size >= self.block_points_num:
            # 若采样点的数目大于4096个点，则随机抽取4096个点(replace=False表示不能重复抽取)
            block_points_idxs = np.random.choice(blocks_points_idxs, self.block_points_num, replace=False)
        else:
            # 若采样点的数目小于4096个点，则随机“重复”抽取4096个点(replace=True表示能重复抽取)
            block_points_idxs = np.random.choice(blocks_points_idxs, self.block_points_num, replace=True)

        # ---------------点特征数据处理---------------
        # 将序号转为点云数据
        blocks_points = scene_points[block_points_idxs, :]  # [N,xyzrgb ]
        # xy平移，使每个block的xy中心点为(0,0)
        translation_xyz = np.zeros((self.block_points_num, 3))  # 聪明：申请了一个新内存空间，避免了np的浅拷贝问题
        translation_xyz[:, 0] = blocks_points[:, 0] - center[0]
        translation_xyz[:, 1] = blocks_points[:, 1] - center[1]
        translation_xyz[:, 2] = blocks_points[:, 2]
        # xyz归一化
        normlized_xyz = np.zeros((self.block_points_num, 3))
        normlized_xyz[:, 0] = blocks_points[:, 0] / scene_coord_max[0]
        normlized_xyz[:, 1] = blocks_points[:, 1] / scene_coord_max[1]
        normlized_xyz[:, 2] = blocks_points[:, 2] / scene_coord_max[2]
        # rgb归一化
        normlized_rgb = np.zeros((self.block_points_num, 3))
        normlized_rgb[:, 0:3] = blocks_points[:, 3:6] / 255.0
        # 数据拼接[x_s,y_s,z, r_n,g_n,b_n,  x_n,y_n,z_n]
        blocks_features = np.concatenate((translation_xyz, normlized_rgb, normlized_xyz), axis=1)

        # 获取对应点云label标签
        blocks_labels = scene_labels[block_points_idxs].astype(int)  # [N]

        # 是否使用数据增强
        if self.transform is not None:
            blocks_features, blocks_labels = self.transform(blocks_features, blocks_labels)
        # ndarry--> current:[N,9] , current_labels[N] ,注意：不要犯傻，B是dataloader加上的
        return blocks_features, blocks_labels

    def __len__(self):
        return len(self.scene_idxs)


# 用于test，需要完整的扫描整个dataset
class ScanAllSceneDataset():
    # prepare to give prediction on each points
    def __init__(self, root, is_train=False, test_area=5, block_points_num=4096, scale=(0.01, 0.01, 0.01), stride=0.5,
                 block_size=1.0,
                 padding=0.001):
        self.root = root  # 数据集root
        self.is_train = is_train  # 区分test和train,优于扫描形式只用于测试，所以默认为test
        self.block_points_num = block_points_num  # 扫描块的点云个数
        self.block_size = block_size  # getitem获得扫描区域块大小，默认为1m*1m
        self.padding = padding  # 扫描填充
        self.stride = stride  # 扫描移动步长0.5
        self.scale = scale  # 原始点云缩放比例，缩放后数据更小推理速度更快
        # 选择加载数据集，加载房间文件的路径
        if self.is_train is True:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) == -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) != -1]
        self.scene_points_num = []  # 房间点云数目 shape:[room_size]
        self.scene_points_list = []  # 点云坐标 shape:[room_size,N,xyzrgb]
        self.semantic_labels_list = []  # 点云标签 shape:[room_size,N]
        self.room_coord_min = []  # 原始点云数据最小值，用于数据平移
        # 加载每个房间的点云数据
        for file in self.file_list:
            data = np.loadtxt(os.path.join(root, file))
            coord_min = np.amin(data[:, :3], axis=0)
            data[:, :3] = data[:, :3] - coord_min[None, :]  # 偏移 [N,3] - [1,3]
            data[:, :3] *= self.scale  # 缩放
            self.room_coord_min.append(coord_min)
            self.scene_points_list.append(data[:, :6])  # xyzrgb
            self.semantic_labels_list.append(data[:, 6])  # label
        assert len(self.scene_points_list) == len(self.semantic_labels_list)
        # 记录所有房间点云点的数目
        for seg in self.semantic_labels_list:
            self.scene_points_num.append(seg.shape[0])

    def __getitem__(self, index):
        # 这里讲按顺序讲点云切分，而不是生成房间索引room_index随机取1m*1m
        # index 表示房间序号，即取第idx房间  Point [N,xyzrgb]（np）
        points = self.scene_points_list[index]
        # 取出该房间label [N]（np）
        labels = self.semantic_labels_list[index]
        # 取出该房间最大最小值
        coord_min, coord_max = np.amin(points[:, :3], axis=0), np.amax(points[:, :3], axis=0)
        # 将room划分为1m*1m的网格
        # grid_x表示x方向扫描次数 grid_x = ((max_x-min_x-block_size)/stride)+1
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)

        # 根据grid生成block，block为我们训练的单个数据
        points_block_array = np.array([])
        label_block_array = np.array([])
        index_block_array = np.array([])
        # 根据grid_x，grid_y，stride，block_size划分房间
        for index_y in tqdm(range(0, grid_y), desc="DATA CONVERSION", colour="BLUE"):
            for index_x in range(0, grid_x):
                # ---------------1*1m区域点云筛选+补充-------------
                # s_xy-------
                #     |1m*1m|
                #     -------e_xy
                # s_x即source_x，取1*1区域原点x
                s_x = coord_min[0] + index_x * self.stride
                # e_x即end_x，取1*1区域最远处x
                e_x = min(s_x + self.block_size, coord_max[0])
                # 修正x，最后不足1m位置修正后的s_x到e_x的距离可能小于1m)
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                # 筛选1*1区域内的点云，padding = 0.001 将点云进行了扩展
                blocks_points_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                            points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                # 如果当前位置中没有点则跳出循环
                if blocks_points_idxs.size == 0:
                    continue
                # 计算1m*1m区域的点云可以分为多少次采样(独立测试数据)，ceil表示向上取整，保证数据完整性
                blocks = int(np.ceil(blocks_points_idxs.size / self.block_points_num))
                # point_size 为进行num_batch 采样后点的数量
                point_size = int(blocks * self.block_points_num)
                # replace表示随机抽取是否可重复
                # 例如:原始点云 size =1 差4095，补齐4096这就要求重复抽取
                #     原始点云 size =3000 差 1096，补齐4096就要求不可重复抽取
                replace = False if ((point_size - blocks_points_idxs.size) <= blocks_points_idxs.size) else True
                # 是否可重复的从 原始数据中抽取缺少部分凑齐4096
                point_idxs_repeat = np.random.choice(blocks_points_idxs, point_size - blocks_points_idxs.size,
                                                     replace=replace)
                # 拼接，将随机抽取+原始点云拼接
                blocks_points_idxs = np.concatenate((blocks_points_idxs, point_idxs_repeat))
                # 同一个 1*1 区域内打乱序号顺序
                np.random.shuffle(blocks_points_idxs)

                # 提示1：并没有将1*1m 区域多次采样的block数据分开，而是按顺序保存，因为同属1m*m区域，我们只要按照顺序依次读取block_num即可，相当于训练多次该1*1m
                # 提示2：并没没有将多个1*1m的区域分开，而是按顺序保存，因为1m*1m区域独立处理成block_num个，读取时按顺序都是独立测试样本

                # ---------------点特征数据处理---------------
                # 将序号转为点云数据
                blocks_points = points[blocks_points_idxs, :]  # [N,xyzrgb ]
                # xy平移，使每个block的xy中心点为(0,0)
                translation_xyz = np.zeros((point_size, 3))  # 聪明：申请了一个新内存空间，避免了np的浅拷贝问题
                translation_xyz[:, 0] = blocks_points[:, 0] - (s_x + self.block_size / 2.0)
                translation_xyz[:, 1] = blocks_points[:, 1] - (s_y + self.block_size / 2.0)
                translation_xyz[:, 2] = blocks_points[:, 2]
                # xyz归一化
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = blocks_points[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = blocks_points[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = blocks_points[:, 2] / coord_max[2]
                # rgb归一化
                normlized_rgb = np.zeros((point_size, 3))  # 聪明：申请了一个新内存空间，避免了np的浅拷贝问题
                normlized_rgb[:, 0:3] = blocks_points[:, 3:6] / 255.0
                # 数据拼接[x_s,y_s,z, r_n,g_n,b_n,  x_n,y_n,z_n]
                blocks_features = np.concatenate((translation_xyz, normlized_rgb, normlized_xyz), axis=1)

                # 获取对应点云label标签
                blocks_labels = labels[blocks_points_idxs].astype(int)  # [N]

                # [N,xyzrgb]
                # 注意此时调整完

                # 该房间点云特征，在第0维度对数据进行拼接[N1,xyzrgb]+[N2,xyzrgb]=[N1+N2,xyzrgb]
                points_block_array = np.vstack(
                    [points_block_array, blocks_features]) if points_block_array.size else blocks_features
                # 该房间点云标签，在第1维度对数据进行拼接[N1]+[N2]-->[N1+N2]
                label_block_array = np.hstack(
                    [label_block_array, blocks_labels]) if label_block_array.size else blocks_labels
                # 该房间点云序号拼接[N1]+[N2]-->[N1+N2]
                # point_idxs 记录的idx 是在原始room中的序号
                index_block_array = np.hstack(
                    [index_block_array, blocks_points_idxs]) if index_block_array.size else blocks_points_idxs

        # 调整通道
        # points_block_array: [N,xyzrgb]-->[block_num,4096,xyzrgb]
        points_block_array = points_block_array.reshape(
            (-1, self.block_points_num, points_block_array.shape[1]))
        # label_block_array: [N]-->[block_num,4096]
        label_block_array = label_block_array.reshape((-1, self.block_points_num))
        # index_block_array: [N]-->[block_num,4096]
        index_block_array = index_block_array.reshape((-1, self.block_points_num))

        """
        data_room :[block_num,4096,xyzrgb]  表示，经过1*1采样后整理所得到的点云顺序，其每4096个点都是在同一个1*1区域
        label_room:[block_num,4096] 表示，data_room的标签
        sample_weight:[block_num,4096] 表示，对应 label_room记录标签的权重
        index_room:[block_num,4096] 表示对应data_room数据在原始点云中的序号位置
        """
        return points_block_array, label_block_array, index_block_array

    def __len__(self):
        # 将每个room作为一个数据集返回，此时len代表 room的数量
        # len一般返回的是数据集的长度，但是test以扫描完一个房间内点为目标，因为每个room的大小不同，划分batch的数量不同，所以先将room返回，在test中挨个划分room的batch
        # 我们的
        return len(self.scene_points_list)


# 用于test，以OneScene作为dataset扫描生成batch，多个场景请循环使用
class ScanOneSceneDataset(Dataset):
    def __init__(self, data_path, scale=(0.01, 0.01, 0.01), block_points_num=4096, stride=0.5, block_size=1.0,
                 padding=0.001):
        """
        data_path: 指定测试数据集，这里为了能完成扫描一个文件，将单个scene看作一个dataste，划分batch
        class_num：类别数目，用于标签权重直方图的计算
        block_points: 一个batch点的数目
        stride :block_points的移动步长
        padding:block可向外扩张长度
        """
        self.block_points_num = block_points_num  # 扫描块的点云个数
        self.block_size = block_size  # getitem获得扫描区域块大小，默认为1m*1m
        self.padding = padding  # 扫描填充
        self.stride = stride  # 扫描移动步长0.5
        self.scale = scale  # 缩放比例
        self.coord_min = None
        self.points = None  # 场景原始点云 xyzrgb [N,6]
        self.semantic_labels = None  # 场景原始点云标签 [N]
        self.points_num = None
        # 加载输入点云数据
        if os.path.exists(data_path):
            data = np.loadtxt(data_path)
            # 偏移
            self.coord_min = np.amin(data[:, :3], axis=0)  # 原始点云数据最小值，用于数据平移 shape [3]
            data[:, :3] = data[:, :3] - self.coord_min[None, :]  # 偏移 [N,3] - [1,3]
            data[:, :3] = data[:, :3] * self.scale  # 缩放
            self.points = data[:, :6]  # 点云 shape:[N,xyzrgb]
            self.semantic_labels = data[:, 6]  # label shape:[N]
            self.points_num = (self.semantic_labels.shape[0])  # 计算场景点的数目
        assert len(self.points) == len(self.semantic_labels)

        # 扫描block
        # 1.生成scan网格
        coord_min, coord_max = np.amin(self.points[:, :3], axis=0), np.amax(self.points[:, :3], axis=0)
        # grid_x表示x方向扫描次数 grid_x = ((max_x-min_x-block_size)/stride)+1
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        # 根据grid生成block，block为我们训练的单个数据
        self.points_block_array = np.array([])
        self.label_block_array = np.array([])
        self.index_block_array = np.array([])
        # 根据grid_x，grid_y，stride，block_size划分房间
        for index_y in tqdm(range(0, grid_y), desc="DATA CONVERSION", colour="BLUE"):
            for index_x in range(0, grid_x):
                # ---------------1*1m区域点云筛选+补充-------------
                # s_xy-------
                #     |1m*1m|
                #     -------e_xy
                # s_x即source_x，取1*1区域原点x
                s_x = coord_min[0] + index_x * self.stride
                # e_x即edge_x，取1*1区域最远处x
                e_x = min(s_x + self.block_size, coord_max[0])
                # 修正x，最后不足1m位置修正后的s_x到e_x的距离可能小于1m)
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                # 筛选1*1区域内的点云，padding = 0.001 将点云进行了扩展
                # 注意：若1*1区域点多，可生成多个block
                blocks_points_idxs = np.where(
                    (self.points[:, 0] >= s_x - self.padding) & (self.points[:, 0] <= e_x + self.padding) & (
                            self.points[:, 1] >= s_y - self.padding) & (
                            self.points[:, 1] <= e_y + self.padding))[0]
                # 如果当前位置中没有点则跳出循环
                if blocks_points_idxs.size == 0:
                    continue
                # 计算1m*1m区域的点云可以分为多少次采样(独立测试数据)，ceil表示向上取整，保证数据完整性
                sub_num_blocks = int(np.ceil(blocks_points_idxs.size / self.block_points_num))
                # point_size 为进行num_batch 采样后点的数量
                point_size = int(sub_num_blocks * self.block_points_num)
                # replace表示随机抽取是否可重复
                # 例如:原始点云 size =1 差4095，补齐4096这就要求重复抽取
                #     原始点云 size =3000 差 1096，补齐4096就要求不可重复抽取
                replace = False if ((point_size - blocks_points_idxs.size) <= blocks_points_idxs.size) else True
                # 是否可重复的从 原始数据中抽取缺少部分凑齐4096
                point_idxs_repeat = np.random.choice(blocks_points_idxs, point_size - blocks_points_idxs.size,
                                                     replace=replace)
                # 拼接，将随机抽取+原始点云拼接
                blocks_points_idxs = np.concatenate((blocks_points_idxs, point_idxs_repeat))
                # 打乱序号顺序
                np.random.shuffle(blocks_points_idxs)

                # 提示1：并没有将1*1m 区域多次采样的block数据分开，而是按顺序保存，因为同属1m*m区域，我们只要按照顺序依次读取block_num即可，相当于训练多次该1*1m
                # 提示2：并没没有将多个1*1m的区域分开，而是按顺序保存，因为1m*1m区域独立处理成block_num个，读取时按顺序都是独立测试样本

                # ---------------点特征数据处理---------------
                # 将序号转为点云数据
                blocks_points = self.points[blocks_points_idxs, :]  # [N,xyzrgb ]
                # xy平移，使每个block的xy中心点为(0,0)
                translation_xyz = np.zeros((point_size, 3))  # 聪明：申请了一个新内存空间，避免了np的浅拷贝问题
                translation_xyz[:, 0] = blocks_points[:, 0] - (s_x + self.block_size / 2.0)
                translation_xyz[:, 1] = blocks_points[:, 1] - (s_y + self.block_size / 2.0)
                translation_xyz[:, 2] = blocks_points[:, 2]
                # xyz归一化
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = blocks_points[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = blocks_points[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = blocks_points[:, 2] / coord_max[2]
                # rgb归一化
                normlized_rgb = np.zeros((point_size, 3))
                normlized_rgb[:, 0:3] = blocks_points[:, 3:6] / 255.0
                # 数据拼接[x_s,y_s,z, r_n,g_n,b_n,  x_n,y_n,z_n]
                blocks_features = np.concatenate((translation_xyz, normlized_rgb, normlized_xyz), axis=1)

                # 获取对应点云label标签
                blocks_labels = self.semantic_labels[blocks_points_idxs].astype(int)  # [N]

                # 注意此时调整完

                # 该房间点云特征，在第0维度对数据进行拼接[N1,xyzrgb]+[N2,xyzrgb]=[N1+N2,xyzrgb]
                self.points_block_array = np.vstack(
                    [self.points_block_array, blocks_features]) if self.points_block_array.size else blocks_features
                # 该房间点云标签，在第1维度对数据进行拼接[N1]+[N2]-->[N1+N2]
                self.label_block_array = np.hstack(
                    [self.label_block_array, blocks_labels]) if self.label_block_array.size else blocks_labels
                # 该房间点云序号拼接[N1]+[N2]-->[N1+N2]
                # point_idxs 记录的idx 是在原始room中的序号
                self.index_block_array = np.hstack(
                    [self.index_block_array, blocks_points_idxs]) if self.index_block_array.size else blocks_points_idxs

        # 调整通道
        # points_block_array: [N,xyzrgb]-->[block_num,4096,xyzrgb]
        self.points_block_array = self.points_block_array.reshape(
            (-1, self.block_points_num, self.points_block_array.shape[1]))
        # label_block_array: [N]-->[block_num,4096]
        self.label_block_array = self.label_block_array.reshape((-1, self.block_points_num))
        # index_block_array: [N]-->[block_num,4096]
        self.index_block_array = self.index_block_array.reshape((-1, self.block_points_num))

    def __getitem__(self, index):
        # index:block编号
        """
        index:block的索引，返回改block的points，label，以及原位置index
        """
        # points_block_array [N,D]
        # label_block_array [N]
        # index_block_array [N]
        return self.points_block_array[index], self.label_block_array[index], self.index_block_array[index]

    def __len__(self):

        return self.points_block_array.shape[0]


if __name__ == '__main__':
    data_root = '../data/stanford_indoor3d_test'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = RandomAllSceneDataset(is_train=True, data_root=data_root, block_points_num=num_point,
                                       test_area=test_area,
                                       block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()
