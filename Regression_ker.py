import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
from CoreNetwork import HWD_LSKNetv1, HWD_LSKNetv2
from module_def import PolyLoss, HuberLoss
from argumentation import argumentation
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import re
from utils import Recorder, metrics, makedir

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


class trainer:
    def __init__(self, args, data_dir, img_name_set, open, et, voa):
        self.model = None
        self.criterion = None
        self.open = open
        self.et = et
        self.voa = voa
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.args = args
        self.data_dir = data_dir
        self.img_name_set = img_name_set
        self.transform = argumentation(112)
        self.get_model()
        self.get_data()
        self.train()
        self.test()

    def get_model(self):
        if self.args.model_name == "v1":
            self.model = HWD_LSKNetv1().to(device)
            self.criterion = HuberLoss(delta=1.4)
        elif self.args.model_name == "v2":
            self.model = HWD_LSKNetv2(self.args.num_classes).to(device)
            self.criterion = PolyLoss(self.args.num_classes)
        else:
            print("wrong!")
        '''if self.args.pretrain:
            #premodel_dir = os.path.join(self.args.model_dir, rf'{self.open}_{self.et}ms_checkpoint.pth')
            premodel_dir = os.path.join(self.args.model_dir, rf'12_500ms_checkpoint.pth')
            self.model.load_state_dict(torch.load(premodel_dir))
            for name, param in self.model.named_parameters():
                if "features.0." in name or "features.2." in name or "features.3." in name or "features.5." in name:
                    param.requires_grad = False'''

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                    weight_decay=1e-7)
        self.StepLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=self.args.gamma)

    def get_data(self):
        dataset = CustomDataset(data_dir=self.data_dir, img_name_set=self.img_name_set, transform=self.transform)

        train_size = int(0.7 * len(dataset))
        test_size = int(0.2 * len(dataset))
        valid_size = int(0.1 * len(dataset))
        train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(dataset,
                                                                                   [train_size, test_size,
                                                                                    valid_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.args.val_batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.val_batch_size, shuffle=False)

    def train(self):
        recorder = Recorder.Recorder(verbose=True)
        makedir.make_dir(self.args.param_dir)
        best_model_path = os.path.join(self.args.param_dir, rf'{self.open}_{self.voa}_{self.et}ms_checkpoint.pth')
        for epoch in range(self.args.epochs):
            self.model.train()
            train_loss = []

            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                outputs, labels = outputs.to(torch.float32), labels.to(torch.float32)
                loss = self.criterion(outputs, labels)

                train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.StepLR.step()

            train_loss = np.average(train_loss)

            if (epoch) % self.args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.validate(self.valid_loader)
                recorder(vali_loss, self.model, best_model_path)
            print(f'Opening {self.open} Voa {self.voa}db et {self.et} Epoch [{epoch+1}/{self.args.epochs}], Training Loss: {train_loss:.4f}')

        #best_model_path = self.args.param_dir + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def validate(self, valid_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                outputs, labels = outputs.to(torch.float32), labels.to(torch.float32)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [outputs, labels],
                         [preds_lst, trues_lst]))

                loss = self.criterion(outputs, labels)
                total_loss.append(loss.item())

            total_loss = np.average(total_loss)
            preds = np.concatenate(preds_lst, axis=0)
            trues = np.concatenate(trues_lst, axis=0)
            mse, mae, mape = metrics.metric(preds, trues)
        print(f'Validation Loss: {total_loss:.4f}, Validation MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')
        self.model.train()
        return total_loss

    def test(self):
        self.model.eval()
        preds_lst, trues_lst, test_loss = [], [], []
        with torch.no_grad():

            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                outputs, labels = outputs.to(torch.float32), labels.to(torch.float32)
                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [outputs, labels],
                         [preds_lst, trues_lst]))

                loss = self.criterion(outputs, labels)
                test_loss.append(loss.item())


            test_loss = np.average(test_loss)
            preds = np.concatenate(preds_lst, axis=0)
            trues = np.concatenate(trues_lst, axis=0)
            mse, mae, mape = metrics.metric(preds, trues)
            df = pd.DataFrame({
                'Predicted': preds.flatten(),
                'Actual': trues.flatten(),
                'MSE': mse,
                'MAE': mae,
                'MAPE': mape
            })

            makedir.make_dir(self.args.res_dir)
            df.to_excel(os.path.join(self.args.res_dir,fr'{self.open}_{self.voa}db_{self.et}ms_predictions.xlsx'), index=False)
        print(f'Test Loss: {test_loss:.4f}, Test MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')
