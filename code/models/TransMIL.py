import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
# import models.ResNet as ResNet
# from pathlib import Path

try:
    import apex
    apex_available=True
except ModuleNotFoundError:
    # Error handling
    apex_available = False
    pass



class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)

        attention_heads = 8 #8
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//attention_heads, #dim//8
            heads = attention_heads,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.7 #0.1
        )

    def forward(self, x):
        # out= self.attn(self.norm(x))
        out, attn = self.attn(self.norm(x), return_attn=True)
        x = x + out
        # x = x + self.attn(self.norm(x))

        return x, attn


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, in_features, out_features=512):
        super(TransMIL, self).__init__()
        # in_features = 2048
        # inter_features = 1024
        # inter_features_2 = 512
        # out_features = 1024 
        # out_features = 512 
        if apex_available: 
            norm_layer = apex.normalization.FusedLayerNorm
        else:
            norm_layer = nn.LayerNorm

        self.pos_layer = PPEG(dim=out_features)
        # self._fc1 = nn.Sequential(nn.Linear(in_features, int(in_features/2)), nn.GELU(), nn.Dropout(p=0.2), norm_layer(int(in_features/2))) # 2048 -> 1024
        # self._fc1_1 = nn.Sequential(nn.Linear(int(in_features/2), int(in_features/2)), nn.GELU(), nn.Dropout(p=0.2), norm_layer(int(in_features/2))) # 2048 -> 1024
        # self._fc1_2 = nn.Sequential(nn.Linear(int(in_features/2), int(in_features/2)), nn.GELU(), nn.Dropout(p=0.2), norm_layer(int(in_features/2))) # 2048 -> 1024
        # self._fc2 = nn.Sequential(nn.Linear(int(in_features/2), int(in_features/4)), nn.GELU(), nn.Dropout(p=0.2), norm_layer(int(in_features/4))) # 1024 -> 512
        # self._fc3 = nn.Sequential(nn.Linear(int(in_features/4), out_features), nn.GELU()) # 512 -> 256



        if in_features == 2048:
            self._fc1 = nn.Sequential(
                # nn.Linear(in_features, int(in_features/2)), nn.GELU(), nn.Dropout(p=0.6), norm_layer(int(in_features/2)),
                # nn.Linear(int(in_features/2), int(in_features/4)), nn.GELU(), nn.Dropout(p=0.6), norm_layer(int(in_features/4)),
                # nn.Linear(int(in_features/4), out_features), nn.GELU(),
                nn.Linear(in_features, int(in_features/2)), nn.GELU(), nn.Dropout(p=0.6), norm_layer(int(in_features/2)),
                # nn.Linear(int(in_features/2), int(in_features/4)), nn.GELU(), nn.Dropout(p=0.6), norm_layer(int(in_features/4)),
                nn.Linear(int(in_features/2), out_features), nn.GELU(),
                ) 
            
            # self._fc1 = nn.Sequential(
            #     nn.Linear(in_features, int(in_features/2)), nn.GELU(), nn.Dropout(p=0.6), norm_layer(int(in_features/2)),
            #     nn.Linear(int(in_features/2), out_features), nn.GELU(),
            #     ) 
        elif in_features == 1024:
            self._fc1 = nn.Sequential(
                nn.Linear(in_features, int(in_features)), nn.GELU(), nn.Dropout(p=0.2), norm_layer(out_features),
                nn.Linear(in_features, out_features), nn.GELU(), nn.Dropout(p=0.6), norm_layer(out_features)
                ) 
        elif in_features == 768:
            self._fc1 = nn.Sequential(
                nn.Linear(in_features, int(in_features)), nn.GELU(), nn.Dropout(p=0.6), norm_layer(in_features),
                nn.Linear(in_features, out_features), nn.GELU(), nn.Dropout(p=0.6), norm_layer(out_features)
                ) 
        elif in_features == 384:
            self._fc1 = nn.Sequential(
                # nn.Linear(in_features, int(in_features)), nn.GELU(), nn.Dropout(p=0.6), norm_layer(in_features),
                nn.Linear(in_features, out_features), nn.GELU(),
                ) 
        # out_features = 256 
        # self._fc1 = nn.Sequential(
        #     nn.Linear(in_features, out_features), nn.GELU(), nn.Dropout(p=0.2), norm_layer(out_features)
        #     ) 
        # self._fc1_2 = nn.Sequential(nn.Linear(inter_features, inter_features_2), nn.GELU(), nn.Dropout(p=0.5), norm_layer(inter_features_2)) 
        # self._fc1_3 = nn.Sequential(nn.Linear(inter_features_2, out_features), nn.GELU())
        # self._fc1 = nn.Sequential(nn.Linear(in_features, 256), nn.GELU())
        # self._fc1_2 = nn.Sequential(nn.Linear(int(in_features/2), out_features), nn.GELU())
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_features))
        self.n_classes = n_classes
        self.layer1 = TransLayer(norm_layer=norm_layer, dim=out_features)
        self.layer2 = TransLayer(norm_layer=norm_layer, dim=out_features)
        # self.norm = nn.LayerNorm(out_features)
        self.norm = norm_layer(out_features)
        self._fc = nn.Linear(out_features, self.n_classes)

        # self.model_ft = ResNet.resnet50(num_classes=self.n_classes, mlp=False, two_branch=False, normlinear=True).to(self.device)
        # home = Path.cwd().parts[1]
        # # self.model_ft.fc = nn.Identity()
        # # self.model_ft.load_from_checkpoint(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth', strict=False)
        # self.model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
        # for param in self.model_ft.parameters():
        #     param.requires_grad = False
        # self.model_ft.fc = nn.Linear(2048, self.in_features)


    def forward(self, x, return_attn=False): #, **kwargs

        # x = self.model_ft(x).unsqueeze(0)
        # print(x.shape)
        # x = x.unsqueeze(0) # needed for feature extractor Visualization!!!
        # print(x.shape)
        if x.dim() > 3:
            x = x.squeeze(0)
        # elif x.dim() == 3:
        #     x = x.unsqueeze(0)
        h = x.float() #[B, n, 1024]
        h = self._fc1(h) #[B, n, 512]
        # print('Feature Representation: ', h.shape)
        #---->duplicate pad
        H = h.shape[1]
        # print(h.size[1])    
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # _H, _W =   H.sqrt().ceil().int(), H.sqrt().ceil().int(),
        add_length = _H * _W - H

        # print(h.shape)
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        
        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)


        #---->Translayer x1
        # h = self.layer1(h) #[B, N, 512]
        h, attn1 = self.layer1(h) #[B, N, 512]

        # print('After first TransLayer: ', h.shape)

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        # print('After PPEG: ', h.shape)
        
        #---->Translayer x2
        # h = self.layer2(h) #[B, N, 512]
        h, attn2 = self.layer2(h) #[B, N, 512]
        # print('TransMIL attn2: ', attn2.shape) # attn2.shape = [1,8,512,512]

        # print('After second TransLayer: ', h.shape) #[1, 1025, 512] 1025 = cls_token + 1024
        #---->cls_token
        # hh = self.norm(h)
        h = self.norm(h)[:,0]
        # print(h.shape)

        #---->predict
        logits = self._fc(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return logits
        if return_attn:
            return logits, attn2
        else: return logits

if __name__ == "__main__":
    
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(in_features=1024, n_classes=2).cuda()
    # print(model.eval())
    logits, attn = model(data, return_attn=True)
    cls_attention = attn[:,:, 0, :6000]
    values, indices = torch.max(cls_attention, 1)
    mean = values.mean()
    zeros = torch.zeros(values.shape).cuda()
    filtered = torch.where(values > mean, values, zeros)
    
    # filter = values > values.mean()
    # filtered_values = values[filter]
    # values = np.where(values>values.mean(), values, 0)

    print(filtered.shape)
    print(filtered)


    # values = [v if v > values.mean().item() else 0 for v in values]
    # print(values)
    # print(len(values))

    # logits = results_dict['logits']
    # Y_prob = results_dict['Y_prob']
    # Y_hat = results_dict['Y_hat']
    # print(F.sigmoid(logits))
