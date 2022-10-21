import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.7 #0.1
        )

    def forward(self, x):
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
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class IQGM(nn.Module):
    def __init__(self, in_features, n_classes):
        super(IQGM, self).__init__()

        self.in_features = in_features
        self.n_classes = self.n_classes
        self.fc = nn.Linear(self.in_features, self.n_classes)

    def forward(feats):
        c = F.softmax(self.fc(feats))
        _, m_indices = torch.sort(c, 0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0,:]) #critical index?

class MDMIL(nn.Module):
    def __init__(self, n_classes):
        super(MDMIL, self).__init__()
        in_features = 1024
        out_features = 512
        self.pos_layer = PPEG(dim=out_features)
        self._fc1 = nn.Sequential(nn.Linear(in_features, out_features), nn.GELU())
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_features))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=out_features)
        self.layer2 = TransLayer(dim=out_features)
        self.norm = nn.LayerNorm(out_features)
        self._fc2 = nn.Linear(out_features, self.n_classes)


    def forward(self, x): #, **kwargs

        h = x.float() #[B, n, 1024]
        h = self._fc1(h) #[B, n, 512]
        
        # print('Feature Representation: ', h.shape)
        #---->duplicate pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)


        #---->Translayer x1
        h, attn1 = self.layer1(h) #[B, N, 512]

        # print('After first TransLayer: ', h.shape)

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        # print('After PPEG: ', h.shape)
        
        #---->Translayer x2
        h, attn2 = self.layer2(h) #[B, N, 512]

        # print('After second TransLayer: ', h.shape) #[1, 1025, 512] 1025 = cls_token + 1024
        #---->cls_token
        
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        return logits, attn2

if __name__ == "__main__":
    data = torch.randn((1, 6000, 512)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    logits, attn = model(data)
    cls_attention = attn[:,:, 0, :6000]
    values, indices = torch.max(cls_attention, 1)
    mean = values.mean()
    zeros = torch.zeros(values.shape).cuda()
    filtered = torch.where(values > mean, values, zeros)
    
    # filter = values > values.mean()
    # filtered_values = values[filter]
    # values = np.where(values>values.mean(), values, 0)

    print(filtered.shape)


    # values = [v if v > values.mean().item() else 0 for v in values]
    # print(values)
    # print(len(values))

    # logits = results_dict['logits']
    # Y_prob = results_dict['Y_prob']
    # Y_hat = results_dict['Y_hat']
    # print(F.sigmoid(logits))
