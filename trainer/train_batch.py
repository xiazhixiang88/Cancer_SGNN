from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader 
from trainer import Trainer
from parser import parse_optimizer, parse_gnn_model
from lifelines.utils import concordance_index
import pandas as pd
import os
from sklearn.model_selection import KFold
import pickle
from collections import defaultdict
from torch_geometric.data import Data

from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, graph_paths, patient_data, device):
        self.graph_paths = graph_paths
        self.patient_data = patient_data
        self.device = device

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        graph_path = self.graph_paths[idx]
        slide_id, graph_format = os.path.splitext(os.path.basename(graph_path))
        graph_format = graph_format.lstrip('.')

        survival_info = self.patient_data[self.patient_data['slide_id'] == slide_id]
        if survival_info.empty:
            print(f"slide {slide_id} is not in survival list")
            return None

        if graph_format == "pkl":
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f).to(self.device)
        elif graph_format == "pt":
            graph = torch.load(graph_path).to(self.device)
            clean_graph = Data(
                x=graph.x,
                edge_index=graph.edge_index,
            )

        survival_time = survival_info['survival_time'].iloc[0]
        event = int(survival_info['event'])

        survival_label = torch.tensor([survival_time, event], dtype=torch.float32).to(self.device)
        return clean_graph, survival_label


class CoxPHLoss(torch.nn.Module):
    def forward(self, risk_pred, duration, event):
        sorted_indices = torch.argsort(duration, descending=True)
        risk_pred = risk_pred[sorted_indices]
        event = event[sorted_indices]

        hazard_ratio = torch.exp(risk_pred)
        log_cumsum_hazard_ratio = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_pred - log_cumsum_hazard_ratio
        censored_likelihood = uncensored_likelihood * event

        loss = -torch.mean(censored_likelihood)
        return loss


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict, patient_data, slide_num, level):
        super().__init__(config)
        self.loss_fcn = CoxPHLoss()
        self.patient_data = patient_data
        train_path = self.config_data["train_path"]
        test_path = self.config_data["eval_path"]
        all_path = self.config_data["all_path"]

        self.train_paths = self.load_graph_paths(train_path)
        self.test_paths = self.load_graph_paths(test_path)
        self.all_paths = self.load_graph_paths(all_path)
        self.slide_num = slide_num
        self.level = level

    def load_graph_paths(self, path_file):
        with open(path_file, 'r') as f:
            graph_paths = f.read().splitlines()
        return graph_paths

    def extract_patient_id(self, file_path):
        filename = os.path.basename(file_path)
        patient_id = filename.split('-')[0]
        return patient_id

    def load_all_graphs_and_labels(self, graph_paths):
        all_graphs = []
        all_labels = []
        all_ids = []
        for graph_path in graph_paths:
            slide_id, graph_format = os.path.splitext(os.path.basename(graph_path))
            graph_format = graph_format.lstrip('.') 
            
            survival_info = self.patient_data[self.patient_data['slide_id'] == slide_id]
            if survival_info.empty:
                print(f"slide {slide_id} is not in survival list")
                continue
            if graph_format == "pkl":
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f).to(self.device)
            elif graph_format == "pt":
                graph = torch.load(graph_path).to(self.device)
            
            all_ids.append(slide_id)
            survival_time = survival_info['survival_time'].iloc[0]
            event = int(survival_info['event'])
            
            survival_label = torch.tensor([survival_time, event], dtype=torch.float32).to(self.device)
            all_graphs.append(graph)
            all_labels.append(survival_label)
        return all_graphs, all_labels, all_ids

    def train_one_epoch(self, dataloader, fold_index, random_state, fold_num):
        self.optimizer.zero_grad()
        all_risk_preds = []
        all_durations = []
        all_events = []

        for graphs, labels in dataloader:
            risk_preds = self.gnn(graphs).squeeze()

            loss = self.loss_fcn(risk_preds, labels[:, 0], labels[:, 1])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            all_risk_preds.extend(risk_preds.tolist())
            all_durations.extend(labels[:, 0].tolist())
            all_events.extend(labels[:, 1].tolist())

        all_risk_preds = np.array(all_risk_preds)
        all_durations = np.array(all_durations)
        all_events = np.array(all_events)

        c_index = concordance_index(all_durations, -all_risk_preds, all_events)
        mean_risk = np.mean(all_risk_preds)
        train_re = pd.DataFrame({
            'risk_preds': all_risk_preds,
            'durations': all_durations,
            'events': all_events
        })
        output_dir = f"./data/train_results/level{self.level}/slide_num{self.slide_num}/fold_num_{fold_num}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"train_results_seed{random_state}_fold_{fold_index + 1}.csv")
        train_re.to_csv(output_path, index=False)
        print(f"seed{random_state} fold{fold_index + 1} mean risk: {mean_risk}")

        return loss.item(), c_index

    def train(self, train_paths, fold_index, random_state, fold_num) -> None:
        print(f"Start training for fold {fold_index + 1}")
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)
        best_c_index = 0
        best_model_path = self.best_model_path

        dataset = GraphDataset(train_paths, self.patient_data, self.device)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(self.n_epoch):
            self.gnn.train()
            c_index_list = []

            loss, c_index = self.train_one_epoch(dataloader, fold_index, random_state, fold_num)
            c_index_list.append(c_index)

            mean_c_index = np.mean(c_index_list)
            print(f"Fold {fold_index + 1} | Epoch {epoch + 1} | Loss: {loss:.4f} | C-index: {mean_c_index:.4f}")

            if mean_c_index > best_c_index:
                best_c_index = mean_c_index
                torch.save(self.gnn.state_dict(), best_model_path)
                print(f"Fold {fold_index + 1}: New best model saved with C-index: {best_c_index:.4f}")

        print(f"Fold {fold_index + 1} Train sample num: {len(train_paths)}")

    def test(self, patient_data, test_paths, fold_index, random_state, fold_num):
        print(f"Start testing for fold {fold_index + 1}")
        test_graphs, test_labels, all_ids = self.load_all_graphs_and_labels(test_paths)
        best_model_path = self.best_model_path
        self.gnn.load_state_dict(torch.load(best_model_path))
        self.gnn.eval()

        risk_pred_all = []
        durations_all = []
        events_all = []
        risk_pred_dict = {}

        with torch.no_grad():
            for graph, label, wsi_id in zip(test_graphs, test_labels, all_ids):
                risk_pred = self.gnn(graph).squeeze()
                risk_pred_all.append(risk_pred.item())
                durations_all.append(label[0].item())
                events_all.append(label[1].item())
                risk_pred_dict[wsi_id] = [risk_pred.item()]

        print(f"Fold {fold_index + 1} | Test sample num: {len(test_graphs)}")

        patient_data["risk"] = patient_data["slide_id"].map(risk_pred_dict)
        patient_data_filtered = patient_data[patient_data["risk"].notnull()]
        patient_data_expanded = patient_data_filtered.explode("risk", ignore_index=True)

        output_dir = f"./data/test_results/level{self.level}/slide_num{self.slide_num}/fold_num_{fold_num}/test_fold"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_results_seed{random_state}_fold_{fold_index + 1}.csv")
        patient_data_expanded.to_csv(output_path, index=False)
        print(f"Test results for fold {fold_index + 1} saved to {output_path}")

    def k_fold_patient(self, k=2, random_state=42):
        patient_data = self.patient_data
        print(f"Start {k}-fold cross-validation.")
        patient_to_slices = defaultdict(list)
        for path in self.all_paths:
            filename = os.path.basename(path)
            base_name = filename.split(".")[0]
            if '-' in base_name:
                patient_id = base_name.rsplit("-", 1)[0]
            else:
                patient_id = base_name
            patient_to_slices[patient_id].append(path)

        patient_ids = list(patient_to_slices.keys())
        print(f"Patinets number: {len(patient_ids)}")
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for fold, (train_patient_idx, test_patient_idx) in enumerate(kfold.split(patient_ids)):
            print(f"---  {fold + 1}/{k} fold ---")
            train_patient_ids = [patient_ids[i] for i in train_patient_idx]
            test_patient_ids = [patient_ids[i] for i in test_patient_idx]
            train_paths = [path for patient_id in train_patient_ids for path in patient_to_slices[patient_id]]
            test_paths = [path for patient_id in test_patient_ids for path in patient_to_slices[patient_id]]

            print(f"Slides number of train dataset: {len(train_patient_ids)}, Patients number of train dataset: {len(test_patient_ids)}")
            print(f"Slides number of test dataset: {len(train_paths)}, Patients number of test dataset: {len(test_paths)}")

            model_dir = f"./data/model_save/level{self.level}"
            self.best_model_path = f"{model_dir}/best_model_fold_{fold + 1}_seed{random_state}.pt"
            os.makedirs(model_dir, exist_ok=True)
            self.train(train_paths, fold_index=fold, random_state=random_state, fold_num=k)
            self.test(patient_data, test_paths, fold_index=fold, random_state=random_state, fold_num=k)

    def k_fold_slide(self, k=2, random_state=42):
        patient_data = self.patient_data
        print(f"Start {k}-fold cross-validation.")
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.all_paths)):
            print(f"---  {fold + 1}/{k} fold ---")
            train_paths = [self.all_paths[i] for i in train_idx]
            test_paths = [self.all_paths[i] for i in test_idx]
            self.train(train_paths, fold_index=fold, fold_num=k)
            self.test(patient_data, test_paths, fold_index=fold, fold_num=k)

    
