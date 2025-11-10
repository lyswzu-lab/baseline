import json
import os
from glob import glob
from itertools import chain

import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
# from mecla.dataset.factory import register_dataset # <-- 移除这个导入，此项目中不需要

# 确保这个导入路径相对于您的项目结构是正确的
def load_json(file_path):
    with open(file_path, 'rt') as f:
        return json.load(f)

# @register_dataset # <-- 移除这个修饰器
class mimic(Dataset): # <-- 确保继承了 Dataset
    task = 'multilabel' 
    num_labels = 14 
    def __init__(self, root, mode='train', transform=None, phase=None): # 添加 phase 以兼容
        self.root = root
        self.transform = transform
        
        # 兼容 phase 和 mode
        if phase is not None:
             mode = 'train' if phase in ['train', 'trainval'] else 'test'

        # 标签顺序必须与您的预处理脚本(preprocess_chexclusion_parallel.py)一致
        self.classes = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]
        self.weight = None
        self.norm_weight = None

        if mode == 'train':
            print(f"Loading MIMIC train split: train_x.json")
            self.x = load_json(os.path.join(root, 'train_x.json'))
            self.y = load_json(os.path.join(root, 'train_y.json'))
        else: # (mode == 'valid' 或 mode == 'test')
            print(f"Loading MIMIC valid/test split: test_x.json")
            self.x = load_json(os.path.join(root, 'test_x.json'))
            self.y = load_json(os.path.join(root, 'test_y.json'))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        # 路径拼接方式 (根据 image_240a60.png, img_384 在 root 内部)
        image_path = os.path.join(self.root, 'img_384', self.x[item])
        
        # [!! 关键修改 !!] ResNet 需要 3 通道图像
        image = Image.open(image_path).convert('RGB')
        
        label = np.asarray(self.y[item]).astype(np.float32)

        if self.transform:
            image = self.transform(image)
        
        # [!! 关键修改 !!] 返回项目要求的字典格式
        filename = self.x[item]
        return {'image': image, 'target': label, 'name': filename}

    def get_number_classes(self):
        return self.num_labels

# @register_dataset # <-- 移除这个修饰器
class chexpert(Dataset):
    """
    Reference: https://github.com/Optimization-AI/ICCV2021_DeepAUC
    """
    task = 'multilabel'
    num_labels = 5 

    def __init__(self,
                 root='',
                 mode='train',
                 transform=None,
                 phase=None, # 添加 phase
                 class_index=-1,
                 use_frontal=True,
                 use_upsampling=False,
                 flip_label=False,
                 verbose=False,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 **kwargs,
                 ):
        
        # 兼容 phase 和 mode
        if phase is not None:
             mode = 'train' if phase in ['train', 'trainval'] else 'test'

        self.root = root
        self.transform = transform
        self.classes = train_cols
        self.num_labels = len(self.classes)

        # 确定要加载的 .csv 文件的名称
        filename_mode = mode
        if mode == 'test':
            filename_mode = 'test'
        elif mode == 'valid':
            filename_mode = 'test' # 强制 'valid' 模式也加载 'test.csv'
        elif mode == 'train':
            filename_mode = 'train'
        
        csv_path = os.path.join(root, f'{filename_mode}.csv')
        print(f"Loading CheXpert {mode} split from: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # (根据 image_240ab9.png, train/test 文件夹在 root 内部)
        # 移除 'CheXpert-v1.0-small/' 或 'CheXpert-v1.0/' 前缀
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '', regex=False)
        if filename_mode == 'test':
            self.df['Path'] = self.df['Path'].str.replace('valid/', 'test/', regex=False)
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)
        # ( ... 此处省略您提供的 chexpert 中其余的处理逻辑 ... )
        # ( ... upsampling, impute missing values ... )
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                # self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia',
                         'Pneumothorax', 'Pleural Other', 'Fracture', 'Support Devices']:  # other labels
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)
        # ( ... 此处省略您提供的 chexpert 中其余的处理逻辑 ... )

        self._num_images = len(self.df)
        self.select_cols = train_cols
        
        # 修正路径以匹配 image_240ab9.png 的结构
        # 假设 csv 中的 'Path' 已经是 'train/...' 或 'test/...'
        # 如果不是，您可能需要在这里调整
        self._images_list = [os.path.join(root, path) for path in self.df['Path'].tolist()]
        self.targets = self.df[train_cols].values.tolist()


    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        image_path = self._images_list[idx]
        
        # [!! 关键修改 !!] ResNet 需要 3 通道图像
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)

        # [!! 关键修改 !!] 返回项目要求的字典格式
        filename = os.path.basename(image_path)
        return {'image': image, 'target': label, 'name': filename}

    def get_number_classes(self):
        return self.num_labels


# @register_dataset # <-- 移除这个修饰器
class nihchest(Dataset):
    task = 'multilabel'
    num_labels = 14

    def __init__(self, root='', mode='train', transform=None, phase=None): # 添加 phase
        
        self.root = root
        self.transform = transform

        # 兼容 phase 和 mode
        if phase is not None:
             mode = 'train' if phase in ['train', 'trainval'] else 'test'

        # 1. 加载主 CSV 文件
        df_all = pd.read_csv(os.path.join(self.root, 'Data_Entry_2017.csv'))

        # 2. 确定并读取划分文件 (根据 image_240a1d.png)
        if mode == 'train':
            split_filename = 'train_list.txt'
        elif mode == 'valid':
            split_filename = 'val_list.txt' # NIH 有 val_list.txt
        elif mode == 'test':
            split_filename = 'test_list.txt'
        else:
            raise ValueError(f"不支持的 mode: {mode}")

        split_filepath = os.path.join(self.root, split_filename)
        with open(split_filepath, 'rt') as f:
            image_indices_in_split = {x.strip('\n') for x in f.readlines()}

        # 3. 筛选 DataFrame
        df = df_all[df_all['Image Index'].isin(image_indices_in_split)].copy()
        print(f"为模式 '{mode}' 加载了 {len(df)} 条记录 (从 {split_filename})")

        # 4. 构建图像路径 (根据 image_240a1d.png)
        # 优先查找 img_384
        image_folder = 'img_384'
        img_paths = {}
        missing_files = 0
        
        # 备选文件夹列表
        subfolders = [f'images_{i+1:03d}' for i in range(12)]

        for img_index in df['Image Index']:
            expected_path = os.path.join(self.root, image_folder, img_index)
            
            if os.path.exists(expected_path):
                img_paths[img_index] = expected_path
            else:
                # 如果在 img_384 找不到, 尝试在 images_0XX 中找
                found_in_subfolder = False
                for subfolder in subfolders:
                    subfolder_path = os.path.join(self.root, subfolder, 'images', img_index) # 假设原始结构是 'images_0XX/images/...'
                    if os.path.exists(subfolder_path):
                        img_paths[img_index] = subfolder_path
                        found_in_subfolder = True
                        break
                
                if not found_in_subfolder:
                    missing_files += 1
                    img_paths[img_index] = None # 标记为 None

        if missing_files > 0:
            print(f"警告：有 {missing_files} 个图像文件未在 {image_folder} 或 {subfolders} 中找到。")

        df['path'] = df['Image Index'].map(img_paths)
        
        # 移除路径未找到的行
        original_count = len(df)
        df = df.dropna(subset=['path'])
        if len(df) < original_count:
             print(f"警告：因缺少图像文件而最终删除了 {original_count - len(df)} 行 ({mode} 模式)。")

        # 5. 处理标签
        df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
        all_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        all_labels = [x for x in all_labels if len(x) > 0]
        self.classes = sorted(all_labels) # 排序以确保顺序一致
        self.num_labels = len(self.classes)

        for c_label in self.classes:
            if len(c_label) > 1:
                df[c_label] = df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

        # 6. 设置最终数据
        self.x = df['path'].values.tolist()
        self.y = df[self.classes].values.astype(np.float32)

        # 7. 计算权重 (保持不变)
        if len(self.y) > 0:
            pos_counts = np.sum(self.y, axis=0)
            neg_counts = len(self.y) - pos_counts
            weight_pos = np.divide(1.0, pos_counts, out=np.zeros_like(pos_counts, dtype=float), where=pos_counts!=0)
            weight_neg = np.divide(1.0, neg_counts, out=np.zeros_like(neg_counts, dtype=float), where=neg_counts!=0)
            self.weight = np.stack([weight_neg, weight_pos], axis=1)

            norm_denom = np.sqrt(np.sum(np.sum(self.y, axis=0)**2))
            self.norm_weight = np.divide(np.sum(self.y, axis=0), norm_denom, out=np.zeros_like(np.sum(self.y, axis=0), dtype=float), where=norm_denom!=0)
        else:
             print(f"警告：未找到模式 '{mode}' 的有效数据。无法计算权重。")
             self.weight = np.ones((len(self.classes), 2)) if self.classes else np.array([])
             self.norm_weight = np.zeros(len(self.classes)) if self.classes else np.array([])


    def __getitem__(self, idx):
        img_path = self.x[idx]
        image = Image.open(img_path).convert('RGB') # 确保是 RGB
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)
        
        # [!! 关键修改 !!] 返回项目要求的字典格式
        filename = os.path.basename(img_path)
        return {'image': image, 'target': label, 'name': filename}

    def get_number_classes(self):
        return self.num_labels