from globals import *
import yaml
import argparse
import random
import torch
from utils_local import ordered_yaml
import pandas as pd
from datetime import datetime
import glob
from sklearn.model_selection import train_test_split
import os
# from trainer.train_gnn_kfold import GNNTrainer
from trainer.train_batch import GNNTrainer
from trainer.train_cnn_kfold import ResNetTrainer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=42)

args = parser.parse_args()

opt_path = args.config

model = "gnn"
if model == "gnn":
    default_config_path = "training_config/survival_analysis.yml"
elif model == "cnn":
    default_config_path = "SUR/Resnet_SUR.yml"

if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path

# Set seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)

mode = "train"
####SUR#####
patient_data = pd.read_excel("data/OS.xlsx")
patient_data['patient_id'] = patient_data['patient_id'].astype(str)
patient_data = patient_data[(patient_data["dead_date"] != "O P") & (patient_data["dead_date"] != "N id")]
end_date = datetime.strptime("2024-08-03", "%Y-%m-%d")
patient_data['Diag_date'] = pd.to_datetime(patient_data['Diag_date'], errors='coerce')
patient_data['dead_date'] = patient_data['dead_date'].apply(
    lambda x: end_date if isinstance(x, str) and "sur" in x else (x if isinstance(x, datetime) else pd.to_datetime(x, errors='coerce'))
)
patient_data['survival_time'] = (patient_data['dead_date'] - patient_data['Diag_date']).dt.days
patient_data['event'] = patient_data['dead_date'] != end_date
patient_data = patient_data.dropna(subset=['survival_time'])

mult = pd.read_csv("data/Multiclass.csv")
mult['label'] = mult['label'].replace({"wei":1, "zhichang":1, "yuanfa":0})

merged = patient_data.merge(mult, left_on='patient_id', right_on='case_id', how='left')

patient_data = merged[merged['label']==0]

patient_data.to_csv("patient_data.csv", index=False)

print(f"data num: {patient_data.shape[0]}")

level = 2

if level == 2:
    all_paths = glob.glob("data/create_save/graph_files/*.pt")
elif level == 1:
    all_paths = glob.glob("data/create_save_level1/graph_files/*.pt")
elif level == 0:
    all_paths = glob.glob("data/create_save_level0/graph_files/*.pt")


def Cancer_Survival_train_val(all_paths, patient_data):
    patient_ids = set(patient_data['patient_id'].astype(str))
    print(f"data num: {len(patient_ids)}")
    
    train_set, test_set = train_test_split(patient_ids, test_size=0.5, random_state=42)

    def save_to_txt(file_path, data_set):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for path in data_set:
                f.write(f"{path}\n")
    save_to_txt("./data/SUR_hover_lv1/list_survival_f1/yuedix5_train.txt", train_set)
    save_to_txt("./data/SUR_hover_lv1/list_survival_f1/yuedix5_test.txt", test_set)
    save_to_txt("./data/SUR_hover_lv1/list_survival_f1/yuedix5_all.txt", filtered_paths)
    return len(patient_ids)

train_val = True
if train_val:
    slide_num = Cancer_Survival_train_val(all_paths, patient_data)

def main():
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")
    if mode == "train":
        # seed_list = range(100) # 100 random experiments were conducted to examine the performance distribution of each model
        seed_list = [random.randint(0, 99)] # Random seed for spliting dataset
        if config["train_type"] == "gnn":
            trainer = GNNTrainer(config, patient_data, slide_num, level)
        elif config["train_type"] == "cnn":
            trainer = ResNetTrainer(config, patient_data)
        else:
            raise NotImplementedError("This type of model is not implemented")
        for random_state in tqdm(seed_list):
            trainer.k_fold_patient(k=4, random_state=random_state)
        # trainer.train()
        # trainer.test(patient_data)
    elif mode == "eval":
        if config["eval_type"] == "homo-graph":
            evaluator = HomoGraphEvaluator(config)
        else:
            raise NotImplementedError("This type of evaluator is not implemented")
        evaluator.eval()
    elif mode == "graph_explain":
        explainer = ExplainGraph(config)
        explainer.eval()
    elif mode == "construct_graph":
        pass


if __name__ == "__main__":
    main()


