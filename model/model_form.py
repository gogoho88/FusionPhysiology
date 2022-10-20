"""
- Network model architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sub_model_TransFusor import GPT

class _Conv_block_2D(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, activation, dim, Dropout=False):
        super(_Conv_block_2D, self).__init__()
        layers = []
        H, W = dim
        self.block = nn.Sequential(
                        nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding, bias=False),
                        # nn.LayerNorm([outplanes, H, W])
                        nn.BatchNorm2d(outplanes)
        )
        layers = [self.block]
        if activation=='tanh':
            layers.append(nn.Tanh())
        elif activation=='relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation=='Lrelu':
            layers.append(nn.LeakyReLU(0.2,inplace=True))
        if Dropout>0:
            layers.append(nn.Dropout2d(p=Dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        x = self.block(x)
        return x

class _Conv_block_2D_TSM(nn.Module):
    """
    - https://github.com/xliucs/MTTS-CAN/blob/main/code/model.py
    - https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py
    """
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, activation, dim, n_segment, fold_div=3, Dropout=False):
        super(_Conv_block_2D_TSM, self).__init__()
        self.block_conv = _Conv_block_2D(inplanes, outplanes, kernel_size, stride, padding, activation, dim, Dropout)
        self.n_segment = n_segment
        self.fold_div = fold_div
    
    def forward(self, x):
        x = self.TemporalShift(x, self.n_segment, self.fold_div)
        x = self.block_conv(x)
        return x

    @staticmethod
    def TemporalShift(x, n_segment, fold_div=3):
        nt, C, H, W = x.size()
        n_batch = nt //n_segment
        x = x.view(n_batch,n_segment,C,H,W)

        fold = C//fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt,C,H,W)

class Attention_mask(nn.Module):
    def __init__(self,Cin):
        super(Attention_mask, self).__init__()
        self.conv_1x1 = nn.Sequential(nn.Conv2d(Cin, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                    nn.Sigmoid())

    def forward(self,x):
        _,_,H,W = x.size()
        x = self.conv_1x1(x)
        xsum = torch.sum(abs(x),(2,3),keepdim=True)
        return (H*W*x)/(2*xsum)

# %% MainNet for Video (MTTS_CAN)
class main_Net_MTTS_CAN(nn.Module):
    """
    - refer/modified form https://github.com/xliucs/MTTS-CAN/blob/main/code/model.py
    """
    def __init__(self, dim_frame=[30,36,36], channel=[32,32,64,64,128], drop_rate=[0.25,0.5], fold_div=3):
        super(main_Net_MTTS_CAN, self).__init__()
        self.len_frame,self.H,self.W = dim_frame
        self.channel = channel
        self.drop_rate = drop_rate
        self.fold_div = fold_div
        self.c12_A = nn.Sequential(
                        _Conv_block_2D(3, channel[0], kernel_size=3, stride=1, padding=1, activation='tanh'),
                        _Conv_block_2D(channel[0], channel[1], kernel_size=3, stride=1, padding=1, activation='tanh'))
        self.c12_M = nn.Sequential(
                    _Conv_block_2D_TSM(3, channel[0], kernel_size=3, stride=1, padding=1, activation='tanh', n_segment=self.len_frame, fold_div=self.fold_div),
                    _Conv_block_2D_TSM(channel[0], channel[1], kernel_size=3, stride=1, padding=1, activation='tanh', n_segment=self.len_frame, fold_div=self.fold_div))
        self.atten1 = Attention_mask(Cin=channel[1])
        self.p1_drop_A = nn.Sequential(nn.AvgPool2d(2,2),
                                        nn.Dropout2d(p=self.drop_rate[0]))
        self.p1_drop_M = nn.Sequential(nn.AvgPool2d(2,2),
                                        nn.Dropout2d(p=self.drop_rate[0]))
        self.c34_A = nn.Sequential(
                        _Conv_block_2D(channel[1], channel[2], kernel_size=3, stride=1, padding=1, activation='tanh'),
                        _Conv_block_2D(channel[2], channel[3], kernel_size=3, stride=1, padding=1, activation='tanh'))
        self.c34_M = nn.Sequential(
                        _Conv_block_2D_TSM(channel[1], channel[2], kernel_size=3, stride=1, padding=1, activation='tanh', n_segment=self.len_frame, fold_div=self.fold_div),
                        _Conv_block_2D_TSM(channel[2], channel[3], kernel_size=3, stride=1, padding=1, activation='tanh', n_segment=self.len_frame, fold_div=self.fold_div))
        self.atten2 = Attention_mask(Cin=channel[3])
        self.p2_drop_M = nn.Sequential(nn.AvgPool2d(2,2),
                                        nn.Dropout2d(p=self.drop_rate[1]))
        self.FC = nn.Sequential(
                            nn.Linear(self.len_frame*self.channel[3]*self.H//4*self.W//4, self.channel[4]),
                            nn.Tanh(),
                            nn.Dropout(p=self.drop_rate[1]),
                            nn.Linear(self.channel[4], self.len_frame)
        )
        self._initialize_weights()      # parameter initialization
        
    def forward(self, vidA, vidM):
        B, T, C, H, W = vidM.size()     # Batch x Time x Channel x Height x Width
        vidA = vidA.mean(axis=1)
        vidM = vidM.contiguous().view(B*T,C,H,W)

        vidA = self.c12_A(vidA)
        vidM = self.c12_M(vidM)
        vidA_mask = self.atten1(vidA).unsqueeze(1)
        vidM = self.attention_masking(vidM,vidA_mask,B,T,self.channel[1],H,W)           #
        vidA = self.p1_drop_A(vidA)
        vidM = self.p1_drop_M(vidM)

        vidA = self.c34_A(vidA)
        vidM = self.c34_M(vidM)
        vidA_mask = self.atten2(vidA).unsqueeze(1)    
        vidM = self.attention_masking(vidM,vidA_mask,B,T,self.channel[3],H//2,W//2)     #
        vidM = self.p2_drop_M(vidM)

        vidM = vidM.view(B,T,self.channel[3],H//4,W//4)
        vidM = vidM.flatten(start_dim=1)                                                #
        vital = self.FC(vidM)

        return vital
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)    # initialization for 'Tanh' activation module
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d)) or isinstance(m, (nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def attention_masking(x,mask,B,T,C,H,W):
        x = x.view(B,T,C,H,W)
        x = x*mask
        x = x.view(B*T,C,H,W)
        return x

# %% MainNet for Video (modified ver of MTTS_CAN)
class main_Net_Video(nn.Module):
    """
    - refer/modified form https://github.com/xliucs/MTTS-CAN/blob/main/code/model.py
    - Compared with MTTS-CAN, followings are modified:
    - Add Last layer adaptive pooling (W,H -> 1,1)
    - Add temporal pooling layer (i.e., 3D avg pooling for vidM)
    """
    def __init__(self, dim_frame=[30,36,36], channel=[32,32,64,64,128], drop_rate=[0.25,0.5], fold_div=3):
        super(main_Net_Video, self).__init__()
        self.len_frame,self.H,self.W = dim_frame
        self.channel = channel
        self.drop_rate = drop_rate
        self.fold_div = fold_div
        self.c12_A = nn.Sequential(
                        _Conv_block_2D(3, channel[0], kernel_size=3, stride=1, padding=1, 
                                        activation='tanh', dim=[self.H,self.W]),
                        _Conv_block_2D(channel[0], channel[1], kernel_size=3, stride=1, padding=1, 
                                        activation='tanh', dim=[self.H,self.W])
                        )
        self.c12_M = nn.Sequential(
                    _Conv_block_2D_TSM(3, channel[0], kernel_size=3, stride=1, padding=1, 
                                        activation='tanh', dim=[self.H,self.W], n_segment=self.len_frame, fold_div=self.fold_div),
                    _Conv_block_2D_TSM(channel[0], channel[1], kernel_size=3, stride=1, padding=1, 
                                        activation='tanh', dim=[self.H,self.W], n_segment=self.len_frame, fold_div=self.fold_div))
        self.atten1 = Attention_mask(Cin=channel[1])
        self.p1_drop_A = nn.Sequential(nn.AvgPool2d(2,2),
                                        nn.Dropout2d(p=self.drop_rate[0]))
        self.p1_drop_M = nn.Sequential(nn.AvgPool3d(2,2),   
                                        nn.Dropout3d(p=self.drop_rate[0]))      # 3d pooling (i.e., add temporal pooling)
        self.c34_A = nn.Sequential(
                        _Conv_block_2D(channel[1], channel[2], kernel_size=3, stride=1, padding=1, 
                                        activation='tanh', dim=[self.H//2,self.W//2]),
                        _Conv_block_2D(channel[2], channel[3], kernel_size=3, stride=1, padding=1, 
                                        activation='tanh', dim=[self.H//2,self.W//2]))
        self.c34_M = nn.Sequential(
                        _Conv_block_2D_TSM(channel[1], channel[2], kernel_size=3, stride=1, padding=1,
                                            activation='tanh', dim=[self.H//2,self.W//2], n_segment=self.len_frame//2, fold_div=self.fold_div),
                        _Conv_block_2D_TSM(channel[2], channel[3], kernel_size=3, stride=1, padding=1, 
                                            activation='tanh', dim=[self.H//2,self.W//2], n_segment=self.len_frame//2, fold_div=self.fold_div))
        self.atten2 = Attention_mask(Cin=channel[3])
        self.p2_drop_M = nn.Sequential(nn.AdaptiveAvgPool3d((None,1,1)),
                                        nn.Dropout3d(p=self.drop_rate[1]))      # 3d pooling (i.e., add temporal pooling)
        self.FC = nn.Sequential(
                            nn.Linear(self.len_frame//2*self.channel[3], self.channel[4]),
                            nn.Tanh(),
                            nn.Dropout(p=self.drop_rate[1]),
                            nn.Linear(self.channel[4], self.len_frame)
        )
        self._initialize_weights()      # parameter initialization
        
    def forward(self, vidA, vidM):
        B, T, C, H, W = vidM.size()     # Batch x Time x Channel x Height x Width
        vidA = vidA.mean(axis=1)
        vidM = vidM.contiguous().view(B*T,C,H,W)

        vidA = self.c12_A(vidA)
        vidM = self.c12_M(vidM)
        vidA_mask = self.atten1(vidA).unsqueeze(1)
        vidM = self.attention_masking(vidM,vidA_mask,B,T,self.channel[1],H,W)           #
        vidA = self.p1_drop_A(vidA)
        vidM = vidM.view(B,T,self.channel[1],H,W).permute(0,2,1,3,4).contiguous()                 
        vidM = self.p1_drop_M(vidM)
        vidM = vidM.permute(0,2,1,3,4).contiguous().view(B*T//2,self.channel[1],H//2,W//2)

        vidA = self.c34_A(vidA)
        vidM = self.c34_M(vidM)
        vidA_mask = self.atten2(vidA).unsqueeze(1)    
        vidM = self.attention_masking(vidM,vidA_mask,B,T//2,self.channel[3],H//2,W//2)     #
        vidM = vidM.view(B,T//2,self.channel[3],H//2,W//2).permute(0,2,1,3,4).contiguous()
        vidM = self.p2_drop_M(vidM)

        # vidM = vidM.reshape(B,T,self.channel[3],H//2,W//2)  
        vidM = vidM.flatten(start_dim=1)                                                #
        vital = self.FC(vidM)

        return vital
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)    # initialization for 'Tanh' activation module
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d)) or isinstance(m, (nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def attention_masking(x,mask,B,T,C,H,W):
        x = x.view(B,T,C,H,W)
        x = x*mask
        x = x.view(B*T,C,H,W)
        return x

# %% MainNet for Radar
class main_Net_Radar(nn.Module):
    """
    """
    def __init__(self, dim_frame=[60,128], len_output=30, channel=[32,32,32,32,64,64,128], 
                drop_rate=[0.25,0.5], activation='tanh'):
        super(main_Net_Radar, self).__init__()
        self.F_R, self.T_R = dim_frame
        self.channel = channel
        self.drop_rate = drop_rate
        self.len_output = len_output
        self.activation = activation
        self.c12_R = nn.Sequential(
                        _Conv_block_2D(8, channel[0], kernel_size=5, stride=1, padding=2, 
                        activation=self.activation, dim=[self.F_R,self.T_R]),
                        _Conv_block_2D(channel[0], channel[1], kernel_size=5, stride=1, padding=2, 
                        activation=self.activation, dim=[self.F_R,self.T_R]))
        self.p1_R = nn.AvgPool2d(2,2)
        self.c34_R = nn.Sequential(
                        _Conv_block_2D(channel[1], channel[2], kernel_size=3, stride=1, padding=1, 
                        activation=self.activation, dim=[self.F_R//2,self.T_R//2]),
                        _Conv_block_2D(channel[2], channel[3], kernel_size=3, stride=1, padding=1, 
                        activation=self.activation, dim=[self.F_R//2,self.T_R//2]))
        self.p2_drop_R = nn.Sequential(nn.AvgPool2d(2,2),
                                        nn.Dropout2d(p=self.drop_rate[0]))
        self.c56_R = nn.Sequential(
                        _Conv_block_2D(channel[3], channel[4], kernel_size=3, stride=1, padding=1, 
                        activation=self.activation, dim=[self.F_R//4,self.T_R//4]),
                        _Conv_block_2D(channel[4], channel[5], kernel_size=3, stride=1, padding=1, 
                        activation=self.activation, dim=[self.F_R//4,self.T_R//4]))
        self.p3_drop_R = nn.Sequential(nn.AdaptiveAvgPool2d((1,None)),          # Project in Freq. domain
                                        nn.Dropout2d(p=self.drop_rate[1]))
        self.FC = nn.Sequential(
                            nn.Linear(self.channel[5]*self.T_R//4, self.channel[6]),
                            nn.Tanh(),
                            nn.Dropout(p=self.drop_rate[1]),
                            nn.Linear(self.channel[6], self.len_output)
        )
        self._initialize_weights()      # parameter initialization

    def forward(self, x_STFT):
        x_STFT = self.c12_R(x_STFT)
        x_STFT = self.p1_R(x_STFT)

        x_STFT = self.c34_R(x_STFT)             #
        x_STFT = self.p2_drop_R(x_STFT)

        x_STFT = self.c56_R(x_STFT)             #
        x_STFT = self.p3_drop_R(x_STFT)         
        
        x_STFT = x_STFT.flatten(start_dim=1)    #
        vital = self.FC(x_STFT)

        return vital

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)    # initialization for 'Tanh' activation module
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d)) or isinstance(m, (nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

# %% MainNet for Fusion
class main_Net_Fusion(nn.Module):
    """
    main_Net_Video + main_Net_Radar
    """
    def __init__(self, dim_video=[30,36,36], dim_radar=[128,60], len_output=30, 
                    channel_video=[32,32,64,64,128], channel_radar=[32,32,32,32,64,64,128],
                    drop_rate=[0.25,0.5], activation='tanh', fold_div_video=3):
        super(main_Net_Fusion, self).__init__()
        self.T_V,self.H_V,self.W_V = dim_video
        self.F_R,self.T_R = dim_radar
        self.len_output = len_output
        self.channel_V = channel_video
        self.channel_R = channel_radar
        self.drop_rate = drop_rate
        self.activation = activation
        self.fold_div = fold_div_video

        # L1 + P (Radar)
        self.c12_R = nn.Sequential(
                        _Conv_block_2D(8, self.channel_R[0], kernel_size=5, stride=1, padding=2, 
                                    activation=self.activation, dim=[self.F_R,self.T_R]),
                        _Conv_block_2D(self.channel_R[0], self.channel_R[1], kernel_size=5, stride=1, padding=2, 
                                    activation=self.activation, dim=[self.F_R,self.T_R]))
        self.p1_R = nn.AvgPool2d(2,2)
        # L2_Conv (Video+Radar)
        self.c12_A = nn.Sequential(
                        _Conv_block_2D(3, self.channel_V[0], kernel_size=3, stride=1, padding=1, 
                                    activation=self.activation, dim=[self.H_V,self.W_V]),
                        _Conv_block_2D(self.channel_V[0], self.channel_V[1], kernel_size=3, stride=1, padding=1, 
                                    activation=self.activation, dim=[self.H_V,self.W_V]))
        self.c12_M = nn.Sequential(
                        _Conv_block_2D_TSM(3, self.channel_V[0], kernel_size=3, stride=1, padding=1, 
                                    activation=self.activation, dim=[self.H_V,self.W_V], n_segment=self.T_V, fold_div=self.fold_div),
                        _Conv_block_2D_TSM(self.channel_V[0], self.channel_V[1], kernel_size=3, stride=1, padding=1, 
                                    activation=self.activation, dim=[self.H_V,self.W_V], n_segment=self.T_V, fold_div=self.fold_div))
        self.atten1 = Attention_mask(Cin=self.channel_V[1])
        self.c34_R = nn.Sequential(
                        _Conv_block_2D(self.channel_R[1], self.channel_R[2], kernel_size=3, stride=1, padding=1, 
                                    activation=self.activation, dim=[self.F_R//2,self.T_R//2]),
                        _Conv_block_2D(self.channel_R[2], self.channel_R[3], kernel_size=3, stride=1, padding=1, 
                                    activation=self.activation, dim=[self.F_R//2,self.T_R//2]))
        # P (Video+Radar)
        self.p1_drop_A = nn.Sequential(nn.AvgPool2d(2,2),
                                        nn.Dropout2d(p=self.drop_rate[0]))
        self.p1_drop_M = nn.Sequential(nn.AvgPool3d(2,2),   
                                        nn.Dropout3d(p=self.drop_rate[0]))      # 3d pooling (i.e., add temporal pooling)
        self.p2_drop_R = nn.Sequential(nn.AvgPool2d(2,2),
                                        nn.Dropout2d(p=self.drop_rate[0]))
        # L3_Conv (Video+Radar)
        self.c34_A = nn.Sequential(
                        _Conv_block_2D(self.channel_V[1], self.channel_V[2], kernel_size=3, stride=1, padding=1, 
                                        activation=self.activation, dim=[self.H_V//2,self.W_V//2]),
                        _Conv_block_2D(self.channel_V[2], self.channel_V[3], kernel_size=3, stride=1, padding=1, 
                                        activation=self.activation, dim=[self.H_V//2,self.W_V//2]))
        self.c34_M = nn.Sequential(
                        _Conv_block_2D_TSM(self.channel_V[1], self.channel_V[2], kernel_size=3, stride=1, padding=1, 
                                        activation=self.activation, dim=[self.H_V//2,self.W_V//2], n_segment=self.T_V//2, fold_div=self.fold_div),
                        _Conv_block_2D_TSM(self.channel_V[2], self.channel_V[3], kernel_size=3, stride=1, padding=1, 
                                        activation=self.activation, dim=[self.H_V//2,self.W_V//2], n_segment=self.T_V//2, fold_div=self.fold_div))
        self.atten2 = Attention_mask(Cin=self.channel_V[3])
        self.c56_R = nn.Sequential(
                        _Conv_block_2D(self.channel_R[3], self.channel_R[4], kernel_size=3, stride=1, padding=1, 
                                    activation=self.activation, dim=[self.F_R//4,self.T_R//4]),
                        _Conv_block_2D(self.channel_R[4], self.channel_R[5], kernel_size=3, stride=1, padding=1, 
                                    activation=self.activation, dim=[self.F_R//4,self.T_R//4]))
        # P (Video+Radar)
        self.p2_drop_M = nn.Sequential(nn.AdaptiveAvgPool3d((None,1,1)),
                                        nn.Dropout3d(p=self.drop_rate[1]))      # 3d pooling (i.e., add temporal pooling)
        self.p3_drop_R = nn.Sequential(nn.AdaptiveAvgPool2d((1,None)),          # Project in Freq. domain
                                        nn.Dropout2d(p=self.drop_rate[1]))
        # FC
        self.FC = nn.Sequential(
                            nn.Linear(self.channel_R[5]*self.T_R//4, self.channel_R[6]),
                            # nn.Tanh(),
                            nn.ReLU(True),
                            nn.Dropout(p=self.drop_rate[1]),
                            nn.Linear(self.channel_R[6], self.len_output)
        )
        self.ln1_V = nn.LayerNorm([self.channel_V[1],self.H_V,self.W_V])
        self.ln1_R = nn.LayerNorm([self.channel_R[3],self.F_R//2,self.T_R//2])
        self.ln2_V = nn.LayerNorm([self.channel_V[3],self.H_V//2,self.W_V//2])
        self.ln2_R = nn.LayerNorm([self.channel_R[5],self.F_R//4,self.T_R//4])
        self._initialize_weights()      # parameter initialization
        # Fusion Module
        self.transformer1 = GPT(n_embd=self.channel_V[1],
                                n_head=4,
                                block_exp=4,
                                n_layer=1,
                                T_anchor_dim=self.T_V,
                                seq_len=1,
                                embd_pdrop = 0.1,
                                resid_pdrop = 0.1,
                                attn_pdrop = 0.1,
        )
        self.transformer2 = GPT(n_embd=self.channel_V[3],
                                n_head=4,
                                block_exp=4,
                                n_layer=1,
                                T_anchor_dim=self.T_V//2,
                                seq_len=1,
                                embd_pdrop = 0.1,
                                resid_pdrop = 0.1,
                                attn_pdrop = 0.1,
        )
        

    def forward(self, X_radar, X_vidA, X_vidM):
        """
        Args:
            X_vidA: appearance video_input (tensor): B, T, C, H, W 
            X_vidM: motion video_input (tensor): B, T, C, H, W 
            X_radar: radar_input (tensor): B, C, F, T 
        """
        B_V, T_V, C_V, H_V, W_V = X_vidM.size()     # Batch x Time x Channel x Height x Width
        B_R, C_R, F_R, T_R = X_radar.size()     # Batch x Channel x Freq. x Time
        X_vidA = X_vidA.mean(axis=1)
        X_vidM = X_vidM.contiguous().view(B_V*T_V,C_V,H_V,W_V)
        # L1+P_radar(only)
        X_radar = self.c12_R(X_radar)
        X_radar = self.p1_R(X_radar)
        # L2_video
        X_vidA = self.c12_A(X_vidA)
        X_vidM = self.c12_M(X_vidM)
        X_vidA_mask = self.atten1(X_vidA).unsqueeze(1)
        X_vidM = self.attention_masking(X_vidM,X_vidA_mask,B_V,T_V,self.channel_V[1],H_V,W_V)           #
        # L2_radar
        X_radar = self.c34_R(X_radar)           #
        # Fusion 1
        Fuse_V = self.ln1_V(X_vidM).view(B_V,T_V,self.channel_V[1],H_V,W_V).permute(0,2,1,3,4).mean((3,4))
        Fuse_R = self.ln1_R(X_radar).permute(0,1,3,2).mean(3)
        Fuse_V_transform, Fuse_R_transform = self.transformer1(Fuse_V, Fuse_R)
        Fuse_V_transform = Fuse_V_transform.permute(0,2,1).contiguous().view(B_V*T_V,self.channel_V[1],1,1)
        Fuse_R_transform = Fuse_R_transform.unsqueeze(dim=2)
        X_vidM = Fuse_V_transform + self.ln1_V(X_vidM)
        X_radar = Fuse_R_transform + self.ln1_R(X_radar)
        # P_video
        X_vidA = self.p1_drop_A(X_vidA)
        X_vidM = X_vidM.view(B_V,T_V,self.channel_V[1],H_V,W_V).permute(0,2,1,3,4).contiguous()                 
        X_vidM = self.p1_drop_M(X_vidM)
        X_vidM = X_vidM.permute(0,2,1,3,4).contiguous().view(B_V*T_V//2,self.channel_V[1],H_V//2,W_V//2)
        # P_radar
        X_radar = self.p2_drop_R(X_radar)
        # L3_video
        X_vidA = self.c34_A(X_vidA)
        X_vidM = self.c34_M(X_vidM)
        X_vidA_mask = self.atten2(X_vidA).unsqueeze(1)    
        X_vidM = self.attention_masking(X_vidM,X_vidA_mask,B_V,T_V//2,self.channel_V[3],H_V//2,W_V//2)      #
        # L3_radar
        X_radar = self.c56_R(X_radar)       #
        # Fusion 2
        Fuse2_V = self.ln2_V(X_vidM).view(B_V,T_V//2,self.channel_V[3],H_V//2,W_V//2).permute(0,2,1,3,4).mean((3,4))
        Fuse2_R = self.ln2_R(X_radar).permute(0,1,3,2).mean(3)
        Fuse2_V_transform, Fuse2_R_transform = self.transformer2(Fuse2_V, Fuse2_R)
        Fuse2_V_transform = Fuse2_V_transform.permute(0,2,1).contiguous().view(B_V*T_V//2,self.channel_V[3],1,1)
        Fuse2_R_transform = Fuse2_R_transform.unsqueeze(dim=2)
        X_vidM = Fuse2_V_transform + self.ln2_V(X_vidM)
        X_radar = Fuse2_R_transform + self.ln2_R(X_radar)
        # P_video
        X_vidM = X_vidM.view(B_V,T_V//2,self.channel_V[3],H_V//2,W_V//2).permute(0,2,1,3,4).contiguous()
        X_vidM = self.p2_drop_M(X_vidM)         #
        # P_radar
        X_radar = self.p3_drop_R(X_radar)       #
        # Fuse, FC
        X_vidM = X_vidM.squeeze(3).squeeze(3).flatten(start_dim=1)
        X_radar = X_radar.squeeze(2).flatten(start_dim=1)
        X_Fuse = X_vidM + X_radar
        # X_Fuse = torch.cat([X_vidM,X_radar],dim=1)    # for concat
        vital = self.FC(X_Fuse)
        return vital

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)    
                # nn.init.xavier_normal_(m.weight)    # initialization for 'Tanh' activation module
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d)) or isinstance(m, (nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)    
                # nn.init.xavier_normal_(m.weight)
            
    @staticmethod
    def attention_masking(x,mask,B,T,C,H,W):
        x = x.view(B,T,C,H,W)
        x = x*mask
        x = x.view(B*T,C,H,W)
        return x


# import visdom
# vis = visdom.Visdom()
# temp = F.interpolate(vidA, size=[256,256], mode="bicubic")
# vis.heatmap(temp[0,0])
# for i in range(32):
#     vis.heatmap(temp[0,i]) 
