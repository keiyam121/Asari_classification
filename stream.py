import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

import os
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models.resnet import resnet18

#前処理
transform=transforms.Compose([transforms.ToTensor()])

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

#モデル
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
net.load_state_dict(torch.load('Assari_classification_remBG.pt', map_location=torch.device('cpu')), strict=False)


targets = ['愛知県産', '千葉県産', '中国産','韓国産']

st.title('アサリ分類器')
st.write('愛知県産、千葉県産、中国産、韓国産のアサリを分類します')
uploaded_image=st.file_uploader("ファイルアップロード", type='jpg')





if uploaded_image is not None:
    image=Image.open(uploaded_image,)
    img_array = np.array(image)
    st.image(img_array,use_column_width = None)
   

    # 予測の実行
    if st.button("Predict", key="predict_button"):
        img=transform(image)
        t=net(img.unsqueeze(0))
        y = F.softmax(t)
        prediction=torch.argmax(y).item()
        probability = y.squeeze().tolist()
 
        # 予測結果の表示
        st.write('## Prediction')
        prob=pd.DataFrame([probability],columns=targets)
        st.write(prob)


        # 予測結果の出力
        st.write('## Result')
        st.write('このアサリはきっと',str(targets[int(prediction)]),'です!')