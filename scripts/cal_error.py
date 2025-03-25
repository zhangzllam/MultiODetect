import torch

import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model.CoreNetwork import HWD_LSKNetv1, HWD_LSKNetv2
from model.module_def import PolyLoss, HuberLoss
from argumentation import argumentation
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import re
from utils import Recorder, metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CustomDataset(Dataset):
    def __init__(self, data_dir, img_name_set, transform=None):
        self.data_dir = data_dir
        self.img_name_set = img_name_set
        self.transform = transform

    def __len__(self):
        return len(self.img_name_set)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.img_name_set[idx])  # 图像文件名
        image = Image.open(img_name).convert('RGB')  # 使用PIL加载图像
        # 加载数值目标
        pattern = r'angle_(\d+)_(\d+)db_(\d+)ms_(\d+)hz_.*.png'
        r = re.search(pattern, self.img_name_set[idx])
        target = float(r.group(4))

        if self.transform:
            image = self.transform(image)

        return image, target

class calculation:
    def __init__(self,args, img_name_set, et, open):
        self.model = amount_net().to(device)
        self.transform = argumentation(224)
        self.criterion = HuberLoss(delta=1.8)
        self.args = args
        self.ori_data_dir = os.path.join(args.ori_root, rf'{str(open)}\{str(et)}')
        self.p_data_dir = os.path.join(args.data_root, rf'{str(open)}\{str(et)}')
        self.img_name_set = img_name_set


        self.load_model(et ,open)
        self.get_data()
        self.pridict(et, open)

    def load_model(self, et, open):
        self.model.load_state_dict(torch.load(os.path.join(self.args.param_dir, rf'{open}_{et}ms_checkpoint.pth')))
    def get_data(self):
        dataset_ori = CustomDataset(data_dir=self.ori_data_dir, img_name_set=self.img_name_set, transform=self.transform)
        dataset_p = CustomDataset(data_dir=self.p_data_dir, img_name_set=self.img_name_set, transform=self.transform)

        self.ori_loader = DataLoader(dataset_ori, batch_size=self.args.batch_size, shuffle=False)
        self.p_loader = DataLoader(dataset_p, batch_size=self.args.batch_size, shuffle=False)

    def pridict(self, et, open):
        self.model.eval()
        preds_lst_ori, trues_lst_ori, ori_loss, preds_lst_p, trues_lst_p, p_loss = [], [], [], [], [], []
        with torch.no_grad():
            for images, labels in self.ori_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                outputs, labels = outputs.to(torch.float32), labels.to(torch.float32)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [outputs, labels],
                         [preds_lst_ori, trues_lst_ori]))

                loss = self.criterion(outputs, labels)
                ori_loss.append(loss.item())

            ori_loss = np.average(ori_loss)
            preds_lst_ori = np.concatenate(preds_lst_ori, axis=0)
            trues_lst_ori = np.concatenate(trues_lst_ori, axis=0)

            for images, labels in self.p_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                outputs, labels = outputs.to(torch.float32), labels.to(torch.float32)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [outputs, labels],
                         [preds_lst_p, trues_lst_p]))

                loss = self.criterion(outputs, labels)
                p_loss.append(loss.item())

            p_loss = np.average(p_loss)
            preds_lst_p = np.concatenate(preds_lst_p, axis=0)
            trues_lst_p = np.concatenate(trues_lst_p, axis=0)


            mse_ori, mae_ori, mape_ori = metrics.metric(preds_lst_ori.flatten(), trues_lst_ori.flatten())
            mse_p, mae_p, mape_p = metrics.metric(preds_lst_p.flatten(), trues_lst_p.flatten())
            df = pd.DataFrame({
                'x_ori': trues_lst_ori.flatten(),
                'y_ori': preds_lst_ori.flatten(),
                'x_p':trues_lst_p.flatten(),
                'y_p': preds_lst_p.flatten(),
                'MAE_ori': mae_ori,
                'MAPE_ori': mape_ori,
                'MAE_p': mae_p,
                'MAPE_p': mape_p,
                'error':preds_lst_ori.flatten()-preds_lst_p.flatten()
            })

            # 保存至 Excel 文件
            df.to_excel(fr'.\{open}_{et}ms{self.args.voa}db_error.xlsx', index=False)
        print(f'Ori Loss: {ori_loss:.4f}, Test MSE: {mse_ori:.4f}, P Loss: {p_loss:.4f}, Test MSE: {mse_p:.4f},')









