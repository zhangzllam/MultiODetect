import torch
import re

# Here is the code ：

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def main(img_path, model_path, savefig_rootpath):
    model = torch.load(model_path)
    #print(model)
    target_layers = [model.features[-1]]


    data_transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((112,112))
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    dic = {'1': 0, '10': 1, '15': 2, '2': 3, '20': 4, '30': 5, '40': 6, '5': 7, '50': 8, '60': 9}
    rs_set = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60]
    for rs in rs_set:
        rs_value = dic[str(rs)]
        # Grad CAM
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        # targets = [ClassifierOutputTarget(281)]
        targets = [ClassifierOutputTarget(rs_value)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,
                                          grayscale_cam, use_rgb=True)

        savefig_path = os.path.join(savefig_rootpath,str(rs),"MP")
        plt.imshow(visualization)
        plt.savefig(savefig_path)
        #plt.show()


if __name__ == '__main__':
    opening_set = [1,4,12,36]
    et_set = [100,200,300,400,500,600,700,800,900,1000]
    #rs_set = [1,2,5,10,15,20,30,40,50,60]
    voa = 20
    for opening in opening_set:
        for et in et_set:
            #img_path = r"E:\zzl\rot_speed\testdataset\data" + "\\" + str(opening) + "\\" + str(voa) + "\\" + str(et)
            img_path = rf"E:\zzl\rot_speed\testdataset\data\{str(opening)}\{str(voa)}\{str(et)}\30\angle_{str(opening)}_{str(voa)}db_{str(et)}ms_30hz_1.png"
            savefig_path = r"E:\zzl\rot_speed\CAMresult" + "\\" + str(opening) + "\\" + str(voa) + "\\" + str(et)
            model_path = rf"G:\zzl\CAM\{str(opening)}\{str(voa)}\MP\MP_{str(voa)}_{str(opening)}_{str(et)}ms.pth"
            main(img_path, model_path, savefig_path)












