import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
import glob
from tqdm import tqdm

class CoxPHLoss(nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk_preds, durations, events):
        idx = torch.argsort(durations, descending=True)
        risk_preds = risk_preds[idx]
        events = events[idx]
        log_risk_sum = torch.logcumsumexp(risk_preds, dim=0)
        loss = -torch.sum(risk_preds * events - log_risk_sum * events) / torch.sum(events)
        return loss

class CoxLayer(nn.Module):
    def __init__(self, Xdim=64):
        super(CoxLayer, self).__init__()
        self.beta = nn.Parameter(torch.randn(Xdim))

    def forward(self, x):
        beta = self.beta.to(x.device)
        risk_score = torch.exp(torch.matmul(x, beta))
        return risk_score

class TrainableModel(nn.Module):
    def __init__(self, resnet_layer4, avgpool, mapping_layer, cox_model):
        super(TrainableModel, self).__init__()
        self.resnet_layer4 = resnet_layer4
        self.avgpool = avgpool
        self.mapping_layer = mapping_layer
        self.cox_model = cox_model

    def forward(self, features):
        # Features pass through trainable ResNet part
        features = self.resnet_layer4(features)  # Shape: (batch_size, 512, 1, 1)
        features = self.avgpool(features).view(features.size(0), -1)  # Flatten to (batch_size, 512)
        features = self.mapping_layer(features)  # Map to 64 dimensions
        risk_preds = self.cox_model(features)  # Compute risk
        return risk_preds

class ResNetTrainer:
    def __init__(self, config, patient_data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.loss_fcn = CoxPHLoss()
        self.patient_data = patient_data

        # Load ResNet and split it into two parts
        self.full_resnet = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            self.full_resnet.conv1,
            self.full_resnet.bn1,
            self.full_resnet.relu,
            self.full_resnet.maxpool,
            self.full_resnet.layer1,
            self.full_resnet.layer2,
            self.full_resnet.layer3,
        ).to(self.device)

        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.all_paths = glob.glob(f'{config["datasets"]["patches_path"]}/*')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def save_model(self, path):
        torch.save(self.trainable_model.state_dict(), path)

    def load_model(self, path):
        self.trainable_model.load_state_dict(torch.load(path, map_location=self.device))

    def extract_and_cache_features(self, folder_path, cache_dir="data/cache_dir/ResNet_fixed"):
        cache_path = os.path.join(cache_dir, os.path.basename(folder_path) + ".pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path).to(self.device)

        patch_features = []
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.endswith((".jpeg", ".png"))]

        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feature = self.feature_extractor(image)
                    patch_features.append(feature)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if len(patch_features) == 0:
            raise ValueError(f"No valid images found in {folder_path}")

        patch_features = torch.cat(patch_features, dim=0)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(patch_features.cpu(), cache_path)
        return patch_features

    def remove_leading_zero(self, slide_id):
        if '-' in slide_id:
            parts = slide_id.split('-')
            return f"{parts[0]}-{int(parts[1])}" 
        return slide_id

    def load_all_graphs_and_labels(self, graph_paths):
        all_features = []
        all_labels = []
        all_slide_id = []
        for folder_path in tqdm(graph_paths):
            try:
                cached_features = self.extract_and_cache_features(folder_path)
                wsi_feature, _ = torch.max(cached_features, dim=0)  # Apply pooling
                all_features.append(wsi_feature)
                slide_id = os.path.basename(folder_path)
                slide_id = self.remove_leading_zero(slide_id)
                survival_info = self.patient_data[self.patient_data['slide_id'] == slide_id]
                survival_time = survival_info['survival_time'].iloc[0]
                event = int(survival_info['event'])
                all_labels.append((survival_time, event))
                all_slide_id.append(os.path.basename(folder_path))
            except Exception as e:
                print(f"Error loading data from {folder_path}: {e}")

        all_features = torch.stack(all_features).to(self.device)
        durations = torch.tensor([label[0] for label in all_labels], dtype=torch.float32).to(self.device)
        events = torch.tensor([label[1] for label in all_labels], dtype=torch.float32).to(self.device)
        return all_features, durations, events, all_slide_id

    def train_one_epoch(self, features, durations, events):
        self.trainable_model.train()
        self.optimizer.zero_grad()
        risk_preds = self.trainable_model(features).squeeze()
        loss = self.loss_fcn(risk_preds, durations, events)
        loss.backward()
        self.optimizer.step()

        c_index = concordance_index(
            durations.cpu().numpy(),
            -risk_preds.detach().cpu().numpy(),
            events.cpu().numpy()
        )
        return loss.item(), c_index

    def train(self, train_paths, fold_index):
        print(f"Start training for fold {fold_index + 1}")

        self.trainable_model = TrainableModel(
            resnet_layer4=self.full_resnet.layer4,
            avgpool=self.full_resnet.avgpool,
            mapping_layer=nn.Linear(512, 64).to(self.device),
            cox_model=CoxLayer(Xdim=64).to(self.device),
        ).to(self.device)
        
        # Optimizer for trainable model
        self.optimizer = optim.Adam(self.trainable_model.parameters(), lr=self.config["optimizer"]["lr"])
        
        best_c_index = 0
        best_model_path = f"./data/model_save/best_model_fold_{fold_index + 1}.pth"
        train_features, train_durations, train_events, _ = self.load_all_graphs_and_labels(train_paths)

        for epoch in range(self.config["train"]["num_epochs"]):
            loss, c_index = self.train_one_epoch(train_features, train_durations, train_events)
            print(f"Fold {fold_index + 1} | Epoch {epoch + 1} | Loss: {loss:.4f} | C-index: {c_index:.4f}")

            if c_index > best_c_index:
                best_c_index = c_index
                self.save_model(best_model_path)
                print(f"New best model saved with C-index: {best_c_index:.4f}")

    def test(self, patient_data, test_paths, fold_index):
        print(f"Start testing for fold {fold_index + 1}")
        self.load_model(f"./data/model_save/best_model_fold_{fold_index + 1}.pth")
        self.trainable_model.eval()

        test_features, test_durations, test_events, test_slide_id = self.load_all_graphs_and_labels(test_paths)

        risk_pred_all = []
        risk_pred_dict = {}
        with torch.no_grad():
            risk_preds = self.trainable_model(test_features).squeeze()
            risk_pred_all = risk_preds.cpu().numpy().tolist()

            for slide_id, risk_pred in zip(test_slide_id, risk_pred_all):
                risk_pred_dict[slide_id] = [risk_pred]

        risk_pred_all = np.array(risk_pred_all)
        test_durations = test_durations.cpu().numpy()
        test_events = test_events.cpu().numpy()
        c_index = concordance_index(test_durations, -risk_pred_all, test_events)

        print(f"Fold {fold_index + 1} | Test sample num: {len(test_slide_id)}")
        print(f"Fold {fold_index + 1} | Test C-index: {c_index:.4f}")

        patient_data["risk"] = patient_data["slide_id"].map(risk_pred_dict)
        patient_data_filtered = patient_data[patient_data["risk"].notnull()]
        patient_data_expanded = patient_data_filtered.explode("risk", ignore_index=True)

        output_dir = "./data/test_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_results_fold_{fold_index + 1}.xlsx")
        patient_data_expanded.to_excel(output_path, index=False)
        print(f"Test results for fold {fold_index + 1} saved to {output_path}")
        return c_index

    def k_fold_train_and_eval(self, k=2, random_state=42):
        print(f"Starting {k}-fold cross-validation")
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.all_paths)):
            train_paths = [self.all_paths[i] for i in train_idx]
            test_paths = [self.all_paths[i] for i in test_idx]

            self.train(train_paths, fold_index=fold)
            c_index = self.test(self.patient_data, test_paths, fold_index=fold)
            fold_results.append(c_index)

        print(f"Cross-validation results: {fold_results}")
        print(f"Mean C-index: {np.mean(fold_results):.4f}")
