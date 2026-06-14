import torch
import torch.nn as nn
import math
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import copy
import cv2
import torch.nn.functional as F
from PIL import Image  # Python 3的标准方式
import lpips
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from scipy.ndimage import gaussian_laplace

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
import types

from model_EMG_Net import EMG_Net

# --- 指标计算函数 ---
def calculate_nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def calculate_hfen(gt, pred):
    gt_edge = gaussian_laplace(gt, sigma=1.5)
    pred_edge = gaussian_laplace(pred, sigma=1.5)
    return np.linalg.norm(gt_edge - pred_edge) / np.linalg.norm(gt_edge)


# 初始化 LPIPS 模型
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def calculate_lpips(gt, pred, device):
    img0 = torch.from_numpy(gt).float().to(device).unsqueeze(0).unsqueeze(0)
    img1 = torch.from_numpy(pred).float().to(device).unsqueeze(0).unsqueeze(0)
    img0 = img0.repeat(1, 3, 1, 1) * 2 - 1
    img1 = img1.repeat(1, 3, 1, 1) * 2 - 1
    return loss_fn_vgg(img0, img1).item()

def measure_model_complexity(model, device):

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    print("\n" + "-" * 30)
    print(f"Model Complexity Analysis:")
    print(f"Total FLOPs: {total_flops:.4f} G")
    print(f"Total Params: {total_params:.4f} M")
    print("-" * 30 + "\n")
    return total_flops, total_params

parser = ArgumentParser(description='EMG_Net')

parser.add_argument('--epoch_num', type=int, default=800, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of EMG_Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=5, help='from {5, 10, 20, 30, 40}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--dataset_name', type=str, default='BrainImages', choices=['BrainImages', 'ixi', 'CC359'],
                    help='name of dataset, BrainImages, ixi, CC359')
parser.add_argument('--matrix_name', type=str, default='Radial', choices=['Radial', 'Cartesian', 'Random'],
                    help='name of dataset, Radial, Cartesian, Random')
parser.add_argument('--net', type=str, default='EMG_Net', help='Name of Net')
parser.add_argument('--matrix_dir', type=str, default='Sampling_Masks/Radial', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='../data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

args = parser.parse_args()

epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
matrix_name = args.matrix_name

###########################################################################################

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except:
    pass


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###########################################################################################
Phi_data_Name = '../%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)

Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask']

mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = mask_matrix.to(device)

###########################################################################################

model = EMG_Net(layer_num)
model = model.to(device)

###########################################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/MRI_CS_%s_%s_%s_layer_%d" % (args.model_dir, args.net, args.dataset_name,matrix_name, layer_num)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)),strict=True)
state_dict = model.state_dict()

model.eval()

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.png')


# 保存结果图像
model_name = "%s_%d" % (args.net, layer_num)
result_dir = os.path.join(args.result_dir, test_name,model_name,"cs_ratio_%d" % args.cs_ratio)
print(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


ImgNum = len(filepaths)
PSNR_All, SSIM_All = np.zeros([1, ImgNum]), np.zeros([1, ImgNum])
NMSE_All, HFEN_All, LPIPS_All = np.zeros([1, ImgNum]), np.zeros([1, ImgNum]), np.zeros([1, ImgNum])

Init_PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
Init_SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2 or data.size(-3) == 2
    return (data ** 2).sum(dim=-1).sqrt() if data.size(-1) == 2 else (data ** 2).sum(dim=-3).sqrt()


print('\n')
print("MRI CS Reconstruction Start")

start_time = time()
model.eval()
with torch.no_grad():
    total_runtime=0
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]

        Iorg = cv2.imread(imgName, 0)

        Icol = Iorg.reshape(1, 1, 256, 256) / 255.0

        Img_output = Icol

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        x_in_k_space = torch.fft.fft2(batch_x)
        masked_x_in_k_space = x_in_k_space * mask

        PhiTb = torch.fft.ifft2(masked_x_in_k_space)
        PhiTb = torch.view_as_real(PhiTb).squeeze(1).permute(0, 3, 1, 2)

        start = time()
        x_output, reconstructed_images, M_1, M_2 = model(PhiTb, mask)
        end = time()

        PhiTb = complex_abs(PhiTb)
        x_output = complex_abs(x_output)

        X_rec = np.clip(x_output.cpu().data.numpy().reshape(256, 256), 0, 1).astype(np.float64)
        GT_norm = Iorg.astype(np.float64) / 255.0

        rec_PSNR = psnr(X_rec * 255., Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec * 255., Iorg.astype(np.float64), data_range=255)
        rec_NMSE = calculate_nmse(GT_norm, X_rec)
        rec_HFEN = calculate_hfen(GT_norm, X_rec)
        rec_LPIPS = calculate_lpips(GT_norm, X_rec, device)

        print("[%02d/%02d] Time: %.4fs | PSNR: %.2f | SSIM: %.4f | NMSE: %.4f | HFEN: %.4f | LPIPS: %.4f" %
              (img_no + 1, ImgNum, (end - start), rec_PSNR, rec_SSIM, rec_NMSE, rec_HFEN, rec_LPIPS))

        PSNR_All[0, img_no], SSIM_All[0, img_no] = rec_PSNR, rec_SSIM
        NMSE_All[0, img_no], HFEN_All[0, img_no], LPIPS_All[0, img_no] = rec_NMSE, rec_HFEN, rec_LPIPS

# Final Summary
output_data = "\n" + "=" * 50 + "\n"
output_data += "Summary for %s | CS Ratio: %d\n" % (args.net, args.cs_ratio)
output_data += "Avg PSNR: %.2f, Avg SSIM: %.4f\n" % (np.mean(PSNR_All), np.mean(SSIM_All))
output_data += "Avg NMSE: %.4f, Avg HFEN: %.4f, Avg LPIPS: %.4f\n" % (
np.mean(NMSE_All), np.mean(HFEN_All), np.mean(LPIPS_All))
output_data += "Total Test Time: %.2fs\n" % (time() - start_time)
output_data += "=" * 50 + "\n"

print(output_data)

# Save to Log
if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
output_file_name = "./%s/Full_Metrics_%s_layer_%d.txt" % (args.log_dir, args.net, args.layer_num)
with open(output_file_name, 'a') as f:
    f.write(output_data)

print("MRI CS Reconstruction End")
