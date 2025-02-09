from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import os
from PIL import Image

import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

# インスタンス化
app = FastAPI()

#前処理
transform=transforms.Compose([transforms.ToTensor()])

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

from torchvision.models.resnet import resnet18
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Sequential(nn.Linear(1000, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4))


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

net = Net().cpu().eval()
net.load_state_dict(torch.load('Assari_classification_remBG.pt', map_location=torch.device('cpu'),weights_only=False),strict=False)
@app.post('/predict')
async def make_predictions(file: UploadFile = File(...)):
    contents=await file.read()
    image = Image.open(io.BytesIO(contents))
    img=transform(image)
    t=net(img.unsqueeze(0))
    y = F.softmax(t)
    y_pred=torch.argmax(y).item()
    y_pred_prob = y.squeeze().tolist()
    return {"prediction": y_pred, "probability": y_pred_prob}

 
    
    


    
