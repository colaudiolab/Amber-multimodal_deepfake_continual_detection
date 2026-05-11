## https://github.com/jacobgil/pytorch-grad-cam
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50

# model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]
# input_tensor = # Create an input tensor image for your model..
# # Note: input_tensor can be a batch tensor with several images!

# # We have to specify the target we want to generate the CAM for.
# targets = [ClassifierOutputTarget(281)]

# # Construct the CAM object once, and then re-use it on many images.
# with GradCAM(model=model, target_layers=target_layers) as cam:
#   # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
#   grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#   # In this example grayscale_cam has only one image in the batch:
#   grayscale_cam = grayscale_cam[0, :]
#   visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#   # You can also get the model outputs without having to redo inference
#   model_outputs = cam.outputs
# ---------------------------------------------------------------------------------------------
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
import os

from models.Fine_grained.graph_video_audio_model import GAT_video_audio

import torch
from torch.nn import DataParallel
import numpy as np

class FusionModel(torch.nn.Module):
    def __init__(self, backbone, backbone_output, nb_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.nb_classes = nb_classes
        self.fc = torch.nn.Linear(backbone_output, nb_classes)

    def forward(self, video, audio, train=True, phase=0):
        out = self.backbone(video, audio)
        video_out, audio_out, x = out['video'], out['audio'], out['features']
        x = self.fc(x)
        outputs = {'logits':x, 'video':video_out, 'audio':audio_out}
        return outputs

def load_object(model, file_path: str = None):
    state_dict = torch.load(file_path, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {file_path}")
    return model

@torch.no_grad()
def wrap_data_parallel(model: torch.nn.Module, all_devices, main_device) -> torch.nn.Module:
    if all_devices is not None and len(all_devices) > 1:
        return DataParallel(model, all_devices, output_device = main_device) # type: ignore
    return model

def load_video(fn_img):
    import random
    import numpy as np
    from PIL import Image
    import soundfile as sf
    from torch import Tensor
    import torchvision.transforms as T
    random.seed(0)
    trans = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    num_frame = 4

    temp = fn_img
    ## 随机每10帧内取4帧
    slice_index = np.arange(0, 10, 1)
    random.shuffle(slice_index)
    slice_index = slice_index[:num_frame]
    slice_index.sort()
    slice_index = slice_index.repeat(10)
    slice_index = slice_index.reshape(num_frame, 10).transpose(1, 0)
    a = np.arange(0, 100, 10).reshape(10, 1)
    slice_index = slice_index + a
    # print(slice_index)
    slice_index = slice_index.reshape(-1)

    base = 0
    img_data = []
    global last_img_path
    last_img_path = temp + '/' + str(slice_index[-1]).zfill(5) + '.png'
    for i in range(len(slice_index)):

        fn = temp + '/' + str(slice_index[i] + base).zfill(5) + '.png'

        while i == 0 and not os.path.exists(fn):
            base+=1
            fn = temp + '/' + str(slice_index[i] + base).zfill(5) + '.png'
        try:
            img = Image.open(fn).convert('RGB')
            img = trans(img)
            
            img_data.append(img.unsqueeze(0))
            temp1 = fn
        except:
            print(fn_img+'.'*10)
    img_data = torch.cat(img_data, dim=0)
    img_data = img_data.view(-1, 128, 128)
    # print(img_data.size())

    basename = os.path.basename(fn_img)
    fn_aud = os.path.join(fn_img, basename + ".wav")
    aud_data, _ = sf.read(fn_aud, start=16000, stop=(4)*16000)
    if len(aud_data.shape) == 2:
        aud_data = aud_data[:,0]
    aud_data = Tensor(aud_data)
    if aud_data.size(0) < 16000*4:
        aud_data = torch.cat([aud_data, torch.zeros(16000*4-aud_data.size(0))], dim=0)
    
    return img_data, aud_data

gpus = [4, 5]
all_gpus = [torch.device(f"cuda:{gpu}") for gpu in gpus]
main_device = torch.device(f"cuda:{gpus[0]}")
model = GAT_video_audio()
model = FusionModel(model, 256, 2)
# model = wrap_data_parallel(model, all_gpus, main_device)
# model = load_object(model, file_path="/home/yejianbin/CIL/ACIL/saved_models/GAT_video_audio_MDCDDataset_0.17_TIL/Finetune/2025-07-20T18-20-44/model_1.pth")
model = load_object(model, file_path="/home/yejianbin/CIL/ACIL/saved_models/GAT_video_audio_MDCDDataset_0.17_TIL/Finetune/2025-07-20T18-20-44/model_5.pth")
# model = model.module
model.eval()

target_layers = [model.backbone.cross_att4]
input_tensor = load_video('/mnt/200ssddata2t/yejianbin/LAV-DF/cropped_faces/original_part1/020000')



# =====注册hook start=====
def feature_hook(model, input, output):
    global features
    # x, y = output
    features = output

def extract(g):
    global features_grad
    # x, y = output
    features_grad = g

model._modules.get('backbone')._modules.get('video_encoder')._modules.get('fea_fusion').register_forward_hook(feature_hook)
# =====注册hook end=====

def plot_cam(input_tensor, model, name):
    global features
    global features_grad
    video, audio = input_tensor
    video = video.unsqueeze(0)
    audio = audio.unsqueeze(0)

        # 获取fc层的权重
    outs = model(video, audio)
    video_out, audio_out, logits = outs['video'], outs['audio'], outs['logits']
    prediction = torch.argmax(logits, dim=1)
    pred_class = logits[:, prediction]

    # 获取特征梯度
    features.register_hook(extract)
    pred_class.backward()
    grads = features_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[:,:,0,0].detach().numpy()
    features = features[0].detach().numpy()

    # =====获取CAM start=====
    def makeCAM(feature, weights):
        import cv2
        # batchsize, C, h, w
        bz, h, w = feature.shape
        # (512,) @ (512, 7*7) = (49,)
        print(weights.shape, feature.reshape(bz,-1).shape)
        cam = weights @ (feature.reshape(bz,-1))
        # 归一化到[0, 1]之间
        cam = cam.reshape(h,w)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        # 转换为0～255的灰度图
        cam_gray = np.uint8(255 * cam)
        # 最后，上采样操作，与网络输入的尺寸一致，并返回
        return cv2.resize(cam_gray, (128, 128))

    cam_gray = makeCAM(features, pooled_grads)


    import cv2
    src_image = cv2.imread(last_img_path)
    h, w, _ = src_image.shape
    cam_color = cv2.applyColorMap(cv2.resize(cam_gray, (w, h)), cv2.COLORMAP_HSV)
    cam = src_image * 0.8 + cam_color * 0.2
    # cam = cam_color
    cv2.imwrite(f'/home/yejianbin/tmp/{name}.jpg', cam)

def find_leaf_directories(root_dir, leaf_dirs=None):
    if leaf_dirs is None:
        leaf_dirs = []
    
    for root, dirs, files in os.walk(root_dir):
        # 如果一个目录下没有子目录（dirs为空列表），则将其添加到列表中
        if not dirs:
            leaf_dirs.append(root)
    return leaf_dirs

# leaf_dirs = find_leaf_directories('/mnt/200ssddata2t/yejianbin/DFDCP/cropped_faces/original_videos/')
# i = 0
# for leaf_dir in leaf_dirs:
#     if i > 500:break
#     basename = os.path.basename(leaf_dir)
#     wav = os.path.join(leaf_dir, f"{basename}.wav")
#     input_tensor = load_video(leaf_dir, wav)
#     plot_cam(input_tensor, model, str(i))
#     i+=1

plot_cam(input_tensor, model, str(1))