import torch
import pandas as pd
from model.CoreNetwork import HWD_LSKNetv1, HWD_LSKNetv2
from model.module_def import PolyLoss, HuberLoss
from argumentation import argumentation
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
        img_name = os.path.join(self.data_dir, self.img_name_set[idx])
        image = Image.open(img_name).convert('RGB')
        pattern = r'angle_(\d+)_(\d+)db_(\d+)ms_(\d+)hz_.*.png'
        r = re.search(pattern, self.img_name_set[idx])
        target = float(r.group(4))

        if self.transform:
            image = self.transform(image)

        return image, target

class predicter:
    def __init__(self, args, data_dir, img_name_set, et, open, voa):
        self.model = None
        self.criterion = None
        self.transform = argumentation(112)
        self.args = args
        self.data_dir = data_dir
        self.img_name_set = img_name_set
        self.load_model(et ,open, voa)
        self.get_data()
        self.pridict(et, open, voa)

    def load_model(self, et, open, voa):
        if self.args.model_name == "v1":
            self.model = HWD_LSKNetv1().to(device)
            self.criterion = HuberLoss(delta=1.4)
        elif self.args.model_name == "v2":
            self.model = HWD_LSKNetv2(self.args.num_classes).to(device)
            self.criterion = PolyLoss(self.args.num_classes)
        else:
            print("wrong!")
        self.model.load_state_dict(torch.load(os.path.join(self.args.param_dir, rf'{open}_{voa}_{et}ms_checkpoint.pth')))
    def get_data(self):
        dataset = CustomDataset(data_dir=self.data_dir, img_name_set=self.img_name_set, transform=self.transform)
        self.loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)


    def pridict(self, et, open, voa):
        self.model.eval()
        preds_lst, trues_lst, test_loss = [], [], []
        with torch.no_grad():
            for images, labels in self.loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                outputs, labels = outputs.to(torch.float32), labels.to(torch.float32)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [outputs, labels],
                         [preds_lst, trues_lst]))

                loss = self.criterion(outputs, labels)
                test_loss.append(loss.item())

            test_loss = np.average(test_loss)
            trues_lst = np.concatenate(trues_lst, axis=0)
            preds_lst = np.concatenate(preds_lst, axis=0)
            mse, mae, mape = metrics.metric(preds_lst.flatten(), trues_lst.flatten())

            df = pd.DataFrame({
                'x_ori': trues_lst.flatten(),
                'y_ori': preds_lst.flatten(),
                'MAE': mae,
                'MAPE': mape,

            })


            df.to_excel(fr'.\{open}_{voa}db_{et}ms_error.xlsx', index=False)
        print(f'Loss: {test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')
