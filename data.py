import os
import pickle

import glob
import pandas as pd
import torchvision.transforms
from PIL import Image
import torch
from torch.utils.data import Dataset

import dgl
from dgl.data import DGLDataset
from dgl import transforms


# TODO: Probably need to add to config what transforms we need
transform = transforms.Compose(
    [
        transforms.DropNode(p=0.5),
        transforms.DropEdge(p=0.5),
        transforms.NodeShuffle(),
        transforms.FeatMask(p=0.5, node_feat_names=['feat'])
    ]
)


class WSIData(Dataset):
    def __init__(self, data_root=None):
        self.data_root = data_root
        self.data_list = []
        types = ('*.svs', '*.tif')
        for type_ in types:
            self.data_list.extend(glob.glob(self.data_root + '/**/'+type_, recursive=True))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wsi_path = self.data_list[index]
        return wsi_path


class PatchData(Dataset):
    def __init__(self, wsi_path):
        """
        Args:
            data_24: path to input data
        """
        self.patch_paths = [p for p in wsi_path.glob("*")]
        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.GaussianBlur(kernel_size=3),
            # torchvision.transforms.RandomResizedCrop(size=256),
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):

        img = Image.open(self.patch_paths[idx]).convert('RGB')
        img = self.transforms(img)
        return img


class GraphDataset(DGLDataset):
    def __init__(self, graph_path, normal_path, name_, type_, name='POINTHET'):
        """
        :param data_root: Root of graph
        :param normal_path: Path to the file contain list of normal images
        """
        self.graph_path = graph_path
        self.normal_path = normal_path
        self.name_ = name_
        self.type_ = type_

        super().__init__(name)

    def process(self):

        with open(self.graph_path) as g:
            self.graph_paths = [a.strip() for a in g.readlines()]

        if self.name_ == 'COAD' or self.name_ == 'BRCA':
            with open(self.normal_path) as f:
                # List of path to normal images
                self.normal_list = [l.strip() for l in f.readlines()]

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        graph_path = self.graph_paths[index]

        with open(graph_path, 'rb') as f:
            dgl_graph = pickle.load(f)

        s = str(graph_path)

        if self.name_ == "COAD":
            # COAD training and testing data
            pos = s.find("TCGA")
            label = 0 if s[pos:pos+16] in self.normal_list else 1
        elif self.name_ == "BRCA":
            # BRCA training and testing data
            pos = s.find("TCGA")
            label = 0 if s[pos:pos+16] in self.normal_list else 1
        elif self.name_ == "ESCA":
            # BRCA training and testing data
            pos = s.find("TCGA")
            label = 0 if s[pos:pos+16] in self.normal_list else 1
        else:
            raise ValueError

        if self.type_ == "train":
            dgl_graph = transform(dgl_graph)

        # Add self loop here for homogeneous graphs
        if dgl_graph.is_homogeneous:
            dgl_graph = dgl.add_self_loop(dgl_graph)

        return dgl_graph, label


class C16EvalDataset(DGLDataset):
    def __init__(self, graph_path, annot_path, name='seg'):
        """
        :param data_root: Root of graph
        :param annot_path: Path to the file contain list of normal images
        """
        self.graph_path = graph_path
        self.annot_dir = annot_path

        super().__init__(name)

    def process(self):

        self.graph_paths = []
        self.labels = []
        self.xml_paths = []
        # C16 testing data
        df = pd.read_csv('./data/camelyon16/testing/reference.csv')
        with open(self.graph_path) as g:
            for a in g.readlines():
                a = a.strip()
                head, tail = os.path.split(a)
                label_name = df[(df.NAME == tail[:-4])]['LABEL'].max()
                label = 0 if label_name == 'Normal' else 1

                xml_path = self.annot_dir + tail[:-4] + ".xml"
                if label == 1:
                    self.labels.append(label)
                    self.graph_paths.append(a)
                    self.xml_paths.append(xml_path)

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        graph_path = self.graph_paths[index]
        label = self.labels[index]
        xml_path = self.xml_paths[index]

        with open(graph_path, 'rb') as f:
            dgl_graph = pickle.load(f)

        # Add self loop here for homogeneous graphs
        if dgl_graph.is_homogeneous:
            dgl_graph = dgl.add_self_loop(dgl_graph)

        return dgl_graph, xml_path, label


class TCGACancerStageDataset(DGLDataset):
    def __init__(self, graph_path, label_path, type_, name="tcga_stage"):
        """
        :param data_root: Root of graph
        :param label_path: Path to the file contain list of normal images
        """
        self.graph_path = graph_path
        self.label_path = label_path
        self.type_ = type_

        super().__init__(name)

    def process(self):

        # Make labels
        with open(self.label_path) as f:
            mapping = [l.strip().split(sep="\t") for l in f.readlines()]
            self.mapping = {k: v for k, v in mapping}

        # Read training or testing graphs
        with open(self.graph_path) as g:
            self.graph_paths = [a.strip() for a in g.readlines()]

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        graph_path = self.graph_paths[index]

        with open(graph_path, 'rb') as f:
            dgl_graph = pickle.load(f)

        s = str(graph_path)
        # COAD training and testing data
        pos = s.find("TCGA")
        lb = self.mapping[s[pos:pos + 12]]
        if lb in ['Stage I', 'Stage IA', 'Stage IB']:
            label = 0
        elif lb in ['Stage IIA', 'Stage IIB', 'Stage II', 'Stage IIC']:
            label = 1
        elif lb in ['Stage IIIB', 'Stage IIIC', 'Stage III', 'Stage IIIA']:
            label = 2
        elif lb in ['Stage IV', 'Stage IVA', 'Stage IVB']:
            label = 3
        else:
            raise ValueError("Undefined label")

        if self.type_ == "train":
            dgl_graph = transform(dgl_graph)

        # Add self loop here for homogeneous graphs
        if dgl_graph.is_homogeneous:
            dgl_graph = dgl.add_self_loop(dgl_graph)

        return dgl_graph, label


class TCGACancerTypingDataset(DGLDataset):
    def __init__(self, graph_path, label_path, type_, name="tcga_typing"):
        """
        :param data_root: Root of graph
        :param label_path: Path to the file contain list of normal images
        """
        self.graph_path = graph_path
        self.label_path = label_path
        self.type_ = type_

        super().__init__(name)

    def process(self):

        # Make labels
        with open(self.label_path) as f:
            if "ESCA" in self.label_path:
                mapping = [l.strip().split(sep=",") for l in f.readlines()]
            else:
                mapping = [l.strip().split(sep="\t") for l in f.readlines()]
            self.mapping = {k: v for k, v in mapping}

        # Read training or testing graphs
        with open(self.graph_path) as g:
            self.graph_paths = [a.strip() for a in g.readlines()]

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        graph_path = self.graph_paths[index]

        with open(graph_path, 'rb') as f:
            dgl_graph = pickle.load(f)

        s = str(graph_path)
        # COAD training and testing data
        pos = s.find("TCGA")
        lb = self.mapping[s[pos:pos + 12]]
        if "ESCA" in self.label_path:
            label = int(lb)
        else:
            if lb in ['Infiltrating Ductal Carcinoma']:
                label = 0
            elif lb in ['Infiltrating Lobular Carcinoma']:
                label = 1
            else:
                raise ValueError("Undefined label")

        if self.type_ == "train":
            dgl_graph = transform(dgl_graph)

        # Add self loop here for homogeneous graphs
        if dgl_graph.is_homogeneous:
            dgl_graph = dgl.add_self_loop(dgl_graph)

        return dgl_graph, label
    
from datetime import datetime
class CancerSurvivalAnalysingDataset(DGLDataset):
    def __init__(self, graph_path, patient_data, type_, name="cancer_survival"):
        """
        :param graph_path: Path to the file containing paths of graphs.
        :param patient_data_path: Path to the CSV file with survival data for patients.
        :param type_: Dataset type, e.g., 'train' or 'test'.
        """
        self.graph_path = graph_path
        self.patient_data = patient_data
        self.type_ = type_

        # 设定一个特定的截止日期用于计算生存时间
        self.end_date = datetime.strptime("2024-08-03", "%Y-%m-%d")
        
        super().__init__(name)

    def process(self):
        # 创建一个患者 ID 到生存数据的映射
        self.patient_data.set_index('病理号', inplace=True)

        # 读取图的路径
        with open(self.graph_path) as g:
            self.graph_paths = [line.strip() for line in g.readlines()]

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        # 获取图的路径
        graph_path = self.graph_paths[index]

        dgl_graph = torch.load(graph_path)

        # 从路径中提取患者 ID（假设文件名包含病理号）
        patient_id = self.extract_patient_id(graph_path)

        # 获取对应的生存时间和事件状态
        survival_info = self.patient_data.loc[patient_id]
        
        survival_time = survival_info['生存时间']
        event = int(survival_info['事件'])  # 转为整数，1 表示事件发生，0 表示未发生

        # 将生存标签打包成张量
        survival_label = torch.tensor([survival_time, event], dtype=torch.float32)

        #print(dgl_graph.edges(etype=('0', 'pos', '0')))

        return dgl_graph, survival_label

    def extract_patient_id(self, graph_path):
        """
        从文件路径中提取病理号。假设病理号位于文件路径中的特定位置。
        根据实际数据格式调整提取方法。
        """
        # 示例实现，根据文件名中的位置提取病理号
        s = str(graph_path)
        pos = s.find("B")  # 假设病理号以"B"开头
        patient_id = s[pos:pos + 10]  # 假设病理号长度为10
        return patient_id
