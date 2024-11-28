import torch
import torch.nn as nn


class se_block(nn.Module):
    def __init__(self,channel,reduction = 16):
        super().__init__()
        #AdaptiveAvgPool2d可以指定輸出大小，當參數為1，等於全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            #inplace=True代表允許直接修改輸入數據，節省內存開銷
            #假如為False，ReLU就要先用另一個變數儲存運算結果，然後再賦值給輸入數值
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 

block = se_block(3)
print(block)

