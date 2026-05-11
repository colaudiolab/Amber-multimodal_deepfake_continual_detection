import torch
import torch.nn as nn
from .select_backbone import select_resnet_half


######################################################################################
class My_CNN_RawAud(nn.Module):
    def __init__(self):
        super(My_CNN_RawAud, self).__init__()

        self.netcnnaud_layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=80, stride=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )  # output: batch x 128 x 5991

        self.netcnnaud_layer2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(192),
        )  # output: batch x 192 x 1497

    def forward(self, x):
        x1 = self.netcnnaud_layer1(x)
        x2 = self.netcnnaud_layer2(x1)
        return x2


class LTI_Network(nn.Module):
    def __init__(self, network='resnet18',with_attention=True,
                 temporal_size=40):
        super(LTI_Network, self).__init__()

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32
        self.aud2visspace = None

        self.with_attention = with_attention
        self.temporal_size = temporal_size

        self.netcnnaud = My_CNN_RawAud()

        a_size = [192, 1997]
        v_size = [128, temporal_size, 28, 28]
        stride = int(a_size[1]/v_size[1])
        kernel_size = int(a_size[1] - (v_size[1] - 1) * stride)
        self.netcnnaud_to_vis = nn.Sequential(
                                    nn.Conv1d(a_size[0], v_size[0] // 2, kernel_size=kernel_size, stride=stride),
                                    nn.BatchNorm1d(v_size[0] // 2),
                                    nn.ReLU(),
                                    nn.Conv1d(v_size[0] // 2, v_size[0], kernel_size=3, stride=1, padding=1)
                                )

        self.netcnnlip, self.param = select_resnet_half(network, track_running_stats=False)

        map_size = v_size[1]
        self.final_fc = nn.Sequential(
            nn.Linear(map_size, 1),
            nn.Sigmoid()
        )

        out_emb = v_size[0] // 4
        self.img_emb_layer = nn.Conv1d(v_size[0], out_emb, 1)
        self.aud_emb_layer = nn.Conv1d(v_size[0], out_emb, 1)


        self.fmap_dim = map_size
        self.out_dim = map_size

    def forward_aud(self, x):
        (B, C) = x.shape  # batch, 48000
        x = x.view(B, 1, C)  # batch x 1 x 48000
        x = self.netcnnaud(x)  # batch x 192 x 1497
        x = self.netcnnaud_to_vis(x)  # batch x 128 x 15
        return x


    def forward_lip(self, x):
        (B, _, H, W) = x.shape
        x = x.view(B, 3, -1, H, W)  # batch x 3 x 30 x 224 x 224

        x = self.netcnnlip(x, return_intermediate=False)  # batch x 128 x 15 x 28 x 28
        if self.temporal_size != 15:
            x = nn.functional.adaptive_avg_pool3d(x, (self.temporal_size, 28, 28))

        return x

    def forward(self, vid_seq, aud_seq):
        vid_out = self.forward_lip(vid_seq)  # batch x 128 x 15 x 28 x 28
        aud_out = self.forward_aud(aud_seq)  # batch x 128 x 15

        vid_out = nn.functional.adaptive_avg_pool3d(vid_out, (self.temporal_size, 1, 1))  # batch x 128 x 15 x 1 x 1
        vid_out = vid_out.view(vid_out.shape[0], vid_out.shape[1], vid_out.shape[2])  # batch x 128 x 15
        vid_aud_distance_ = torch.pow((vid_out - aud_out), 2)  # batch x 128 x 15
        fmaps = vid_aud_distance_.clone()
        vid_aud_distance_ = torch.sqrt(torch.sum(vid_aud_distance_, dim=1))  # batch x 15
        

        if self.with_attention:
            img_emb = self.img_emb_layer(vid_out)  # batch x 32 x 15 x 28 x 28 or batch x 32 x 15 (if fine_grained = temporal)
            aud_emb = self.aud_emb_layer(aud_out.view(aud_out.shape[0], aud_out.shape[1], aud_out.shape[2]))  # batch x 32 x 15

            atts = []
            for i_emb, a_emb in zip(img_emb, aud_emb):
                atts.append(torch.sum(i_emb * a_emb, dim=0))  # each, 15

            att = torch.stack(atts)  # batch x 28 x 28 or batch x 15 (temporal) or batch x 15 x 28 x 28 (spatio_temporal)
            att = att / (32)  # normalize with the C size

            att = att.view(att.shape[0], -1)  # batch x 784
            # relaxed softmax
            att = torch.nn.functional.softmax(att, dim=1)

            vid_aud_distance_ = torch.mul(att, vid_aud_distance_)

        else:
            att = None

        # final_out = self.final_fc(vid_aud_distance_)
        
        # return final_out, vid_aud_distance_, att
        vid_out = vid_out.mean(1)
        aud_out = aud_out.mean(1)
        out = {'video': vid_out, 'audio': aud_out, 'features': vid_aud_distance_, 'fmaps': fmaps}
        return out

if __name__ == "__main__":
    net = My_Network()
    logits = net(torch.randn(2, 120, 128, 128), torch.randn(2, 64000))
    print(logits)
    # print(x_video.shape, x_audio.shape, x.shape)
    # print(summary(net, (10, 512)))