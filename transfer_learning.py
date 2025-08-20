#transfer learning
#running single thread in docker with isolcpus and no chrt scheduling explicitly set
#gpu passthrough x2
#link https://www.kaggle.com/competitions/isic-2024-challenge
#put notebook into folder /mnt/zpool1/zpool1/docker_dir/Documents/isic_skin

import os
import numpy as np
import torch
import pandas as pd
import joblib
from sklearnex import patch_sklearn
patch_sklearn()
from PIL import Image
from io import BytesIO
from tqdm import tqdm
PATH = '/mnt/zpool1/zpool1/docker_dir/Documents/isic_skin/train-image.hdf5'
import h5py
train_meta = pd.read_csv('train-metadata.csv')
labels =train_meta[['isic_id', 'target','iddx_1', 'iddx_full']].copy()
train_meta = train_meta.drop(columns =['attribution', 'copyright_license', 'lesion_id', 'iddx_full', 'iddx_1'
, 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence'])
#transform labels into light benign=0, light malignant=2, dark benign=1, dark malignant=3, light premalignant=4
#dark premalignant=5
lookup_table = {
    'Benign': 0,
       'Benign::Benign epidermal proliferations::Lichen planus like keratosis': 0,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, NOS, Junctional': 1,
       'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma in situ':2,
       'Benign::Benign melanocytic proliferations::Nevus':1,
       'Malignant::Malignant adnexal epithelial proliferations - Follicular::Basal cell carcinoma::Basal cell carcinoma, Nodular':2,
       'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma, Invasive':2,
       'Indeterminate::Indeterminate epidermal proliferations::Solar or actinic keratosis':0,
       'Benign::Benign epidermal proliferations::Seborrheic keratosis':0,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, Atypical, Dysplastic, or Clark':1,
       'Malignant::Malignant adnexal epithelial proliferations - Follicular::Basal cell carcinoma::Basal cell carcinoma, Superficial':2,
       'Indeterminate::Indeterminate melanocytic proliferations::Atypical melanocytic neoplasm':1,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma in situ':3,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, NOS, Dermal':1,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma in situ::Melanoma in situ, Lentigo maligna type':3,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive':3,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma, NOS':3,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma in situ::Melanoma in situ, associated with a nevus':3,
       'Malignant::Malignant adnexal epithelial proliferations - Follicular::Basal cell carcinoma::Basal cell carcinoma, Infiltrating':2,
       'Malignant::Malignant adnexal epithelial proliferations - Follicular::Basal cell carcinoma':2,
       'Benign::Benign epidermal proliferations::Pigmented benign keratosis':0,
       'Benign::Benign epidermal proliferations::Solar lentigo':0,
       'Benign::Benign soft tissue proliferations - Fibro-histiocytic::Dermatofibroma':0,
       'Indeterminate::Indeterminate epidermal proliferations::Solar or actinic keratosis::Actinic keratosis, Bowenoid':0,
       'Indeterminate::Indeterminate melanocytic proliferations::Atypical intraepithelial melanocytic proliferation':1,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, NOS, Compound':1,
       'Benign::Cysts::Trichilemmal or isthmic-catagen or pilar cyst':0,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive::Melanoma Invasive, Superficial spreading':3,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive::Melanoma Invasive, Associated with a nevus':3,
       'Benign::Benign soft tissue proliferations - Fibro-histiocytic::Angiofibroma::Angiofibroma, Facial':0,
       'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma, Invasive::Squamous cell carcinoma, Invasive, Keratoacanthoma-type':2,
       'Benign::Benign soft tissue proliferations - Fibro-histiocytic::Scar':0,
       'Benign::Flat melanotic pigmentations - not melanocytic nevus::Lentigo NOS':1,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, Deep penetrating':1,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, Of special anatomic site':1,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma metastasis':3,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, Spitz':1,
       'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma in situ::Squamous cell carcinoma in situ, Bowens disease':2,
       'Benign::Inflammatory or infectious diseases::Verruca':0,
       'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma, NOS':2,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, Combined':1,
       'Benign::Benign melanocytic proliferations::Nevus::Blue nevus::Blue nevus, Cellular':1,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive::Melanoma Invasive, On chronically sun-exposed skin or lentigo maligna melanoma':3,
       'Benign::Benign melanocytic proliferations::Nevus::Nevus, Congenital':0,
       'Benign::Benign soft tissue proliferations - Vascular::Hemangioma::Hemangioma, Cherry':0,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma in situ::Melanoma in situ, Superficial spreading':3,
       'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive::Melanoma Invasive, Nodular':3,
       'Benign::Cysts':0,
       'Benign::Benign adnexal epithelial proliferations - Apocrine or Eccrine::Hidradenoma':0,
       'Benign::Benign soft tissue proliferations - Fibro-histiocytic::Fibroepithelial polyp':0,
       'Benign::Benign adnexal epithelial proliferations - Follicular':0,
       'Benign::Benign epidermal proliferations::Seborrheic keratosis::Seborrheic keratosis, Clonal':0
}

labels['class'] = labels['iddx_full'].map(lookup_table)
#labels['class'] = labels['target']
#print(labels['class'].unique())

train_meta = train_meta.drop(columns =['patient_id', 'target', 'anatom_site_general', 'image_type', 
                                       'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
                                       'tbp_lv_location', 'tbp_lv_location_simple', 'tbp_tile_type'
                                      ])

#drop age and sex because they are not useful, separate id from metadata
train_meta = train_meta.drop(columns =['isic_id', 'sex', 'age_approx'])

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 0.1, cache_size = 2000, class_weight = 'balanced', probability = True)
from sklearn.model_selection import train_test_split as sksplit
from sklearn.metrics import roc_auc_score
target = labels['class']
print(labels['class'].value_counts())
isic_list = labels['isic_id']
train_x, test_x, train_y, test_y, train_id, test_id = sksplit(train_meta, target, isic_list, test_size = 0.05, stratify = target, shuffle = True)
import pandas.api.types
from sklearn.metrics import roc_curve, auc, roc_auc_score


def score(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float=0.80) -> float:
     v_gt = abs(np.asarray(solution.values)-1)
     v_pred = np.array([1.0 - x for x in submission.values])
     max_fpr = abs(1-min_tpr)
     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
     return(partial_auc)


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler

device = 'cuda:0'
train_ids = pd.DataFrame({'labels':train_y, 'ids':train_id})
test_ids = pd.DataFrame({'labels':test_y, 'ids':test_id})

import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from torchvision.transforms import functional as TF

torch.set_default_dtype(torch.float32)
class img_loader():
    def __init__(self, path):
        self.path = path
    def load(isic_id):
        with h5py.File(path, 'r') as f:
            img = f[isic_id][()]
            img = np.frombuffer(img)
        return img
            
def pad_to_size(x: torch.Tensor, dim = 289) -> torch.Tensor:
    assert x.dim() == 3 and x.size(0) == 3, "Expected shape (3, H, W)"
    _, H, W = x.shape

    pad_h = max(0, dim - H)
    pad_w = max(0, dim - W)

    # symmetric (center) padding
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    top_t = torch.zeros((3, top, W))
    bottom_t = torch.zeros((3, bottom, W))
    left_t = torch.zeros((3, dim, left))
    right_t = torch.zeros((3, dim, right))
    x = torch.cat((top_t, x, bottom_t), dim = -2)
    x = torch.cat((left_t, x, right_t), dim = -1)
    return x


class image_load(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = annotations_file
        self.f = h5py.File(img_dir, 'r')
        self.transform = transforms.Compose([transforms.Resize(224),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomHorizontalFlip(p=0.5)
                                            
                                            ])
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        isic_id = self.img_labels.iloc[idx]['ids']
        image = Image.open(BytesIO(self.f[isic_id][()]))
        r,g,b = image.split()
        r = np.array(r, dtype = 'f')
        g = np.array(g, dtype = 'f')
        b = np.array(b, dtype = 'f')
        image = torch.tensor(np.array([r,g,b]), dtype = torch.float32)/255
        #image = pad_to_size(image, 289)
        image = self.transform(image)
        label = self.img_labels.iloc[idx]['labels']
        return image, label

def image_tensor(annotations_file, img_dir):
    img_labels = annotations_file
    f = h5py.File(img_dir, 'r')
    img_list = []
    for i in img_labels['ids']:
        image = Image.open(BytesIO(f[i][()]))
        r,g,b = image.split()
        r = np.array(r, dtype = 'f')
        g = np.array(g, dtype = 'f')
        b = np.array(b, dtype = 'f')
        image = torch.tensor(np.array([r,g,b]), dtype = torch.float32)/255
        image = pad_to_size(image, 289)
        img_list.append(image)
    img_list = torch.tensor(img_list, dtype = torch.float32)
    labels = torch.tensor(img_labels['labels'], dtype = torch.float32)
    return img_list, labels

class MaxCosineNeighborDistance(nn.Module):

    def __init__(self, kernel_size=7, stride=1, padding=3, dilation=1, eps=1e-8):
        super().__init__()
        assert isinstance(kernel_size, int) and kernel_size >= 1, "kernel_size must be a positive int"
        self.k = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size // 2) if padding is None else padding
        self.eps = eps

        # Prepare unfold operator
        self.unfold = nn.Unfold(kernel_size=self.k, dilation=self.dilation,
                                padding=self.padding, stride=self.stride)

        # Precompute neighbor indices (exclude the center position)
        K = self.k * self.k
        center_flat = (self.k // 2) * self.k + (self.k // 2)  # row-major within the window
        idx = torch.arange(K, dtype=torch.long)
        neighbor_idx = torch.cat([idx[:center_flat], idx[center_flat+1:]], dim=0)  # (K-1,)
        self.register_buffer("neighbor_idx", neighbor_idx)
        self.center_flat = int(center_flat)  # Python int for indexing

    def _out_hw(self, H, W):
        H_out = (H + 2*self.padding - self.dilation*(self.k - 1) - 1) // self.stride + 1
        W_out = (W + 2*self.padding - self.dilation*(self.k - 1) - 1) // self.stride + 1
        return H_out, W_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        #assert C == 3, "Expected x with 3 channels (RGB)"
        cols = self.unfold(x)                     # (B, C*K, L) where K=k*k, L=H_out*W_out
        K = self.k * self.k
        L = cols.shape[-1]

        # Reshape to (B, C, K, L)
        patches = cols.view(B, C, K, L)

        # Center pixel vectors: (B, C, L)
        center = patches[:, :, self.center_flat, :]

        # Neighbor pixel vectors: (B, C, K-1, L)
        neighbors = patches.index_select(dim=2, index=self.neighbor_idx)
        dist = ((center.unsqueeze(2) - neighbors)**2).sum(dim=1)
        max_dist = dist.sum(dim =1, keepdim = True) / (8*torch.sqrt(torch.tensor(3)))
        H_out, W_out = self._out_hw(H, W)
        return max_dist.view(B, 1, H_out, W_out)

class LocalVarianceUnfold(nn.Module):
    def __init__(self, kernel_size=8, stride=1, padding=None, dilation=1, reduce_channels=False):
        super().__init__()
        assert isinstance(kernel_size, int) and kernel_size >= 1, "kernel_size must be a positive int"
        self.k = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size // 2) if padding is None else padding
        self.reduce_channels = reduce_channels

        # Unfold over single-channel view to get 1×k×k patches per channel
        self.unfold = nn.Unfold(kernel_size=self.k, dilation=self.dilation,
                                padding=self.padding, stride=self.stride)

    def _out_hw(self, H, W):
        H_out = (H + 2*self.padding - self.dilation*(self.k - 1) - 1) // self.stride + 1
        W_out = (W + 2*self.padding - self.dilation*(self.k - 1) - 1) // self.stride + 1
        return H_out, W_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Treat each channel independently: (B*C, 1, H, W)
        xc = x.contiguous().view(B * C, 1, H, W)

        # Extract k×k patches: (B*C, k*k, L) where L = H_out * W_out
        cols = self.unfold(xc)

        # Variance across window elements (per location), unbiased=False
        mean = cols.mean(dim=1)                  # (B*C, L)
        mean2 = (cols * cols).mean(dim=1)        # (B*C, L)
        var = mean2 - mean * mean                # (B*C, L)

        # Reshape back to spatial map
        H_out, W_out = self._out_hw(H, W)
        var = var.view(B, C, H_out, W_out)

        if self.reduce_channels:
            var = var.mean(dim=1, keepdim=True)  # (B, 1, H_out, W_out)

        return var

class MaxLocaldist(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        assert kernel_size > 1 and isinstance(kernel_size, int), \
            "kernel_size must be an integer > 1"
        self.k = kernel_size
        self.pad = kernel_size // 2
        # We'll use nn.Unfold to extract all sliding patches efficiently
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, H)
        B, C, H, W = x.shape
        assert C == 3 and H == W, "Expected input (B,3,H,H) square images"

        # 1) Extract all k×k patches -> shape (B, 3*k*k, L) where L=(H-k+1)^2
        patches = self.unfold(x)                # (B, C*k*k, L)

        # 2) Reshape to (B, L, k*k, 3)
        L = patches.size(-1)
        patches = patches.view(B, C, self.k*self.k, L)
        patches = patches.permute(0, 3, 2, 1)   # (B, L, N=k*k, C=3)

        # 3) Merge batch & patch dims -> (B*L, N, 3)
        v = patches.reshape(-1, self.k*self.k, C)

        # 4) Compute full pairwise Euclidean distances -> (B*L, N, N)
        dist_mat = torch.cdist(v, v, p=2)

        # 5) Collapse each window to its maximum distance -> (B*L,)
        max_dist = dist_mat.flatten(1).max(dim=1)[0]

        # 6) Reshape back to spatial map (B,1,H-k+1,H-k+1)
        out = max_dist.view(B, 1, H - self.k + 1, W - self.k + 1)

        # 7) Zero-pad to restore original H×H -> (B,1,H,H)
        out = F.pad(out, (self.pad, self.pad, self.pad, self.pad),
                    mode='constant', value=0)
        return 

    
class LocalMean2d(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if not isinstance(kernel_size, int) or kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer")
        self.k = kernel_size
        # We’ll pre-build Unfold with padding=0; we handle asymmetric pad manually.
        self.unfold = nn.Unfold(kernel_size=self.k, dilation=1, padding=0, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {x.shape}")

        B, C, H, W = x.shape
        k = self.k

        # Asymmetric zero padding to keep H, W unchanged for both odd and even k.
        pad_h = k - 1
        pad_w = k - 1
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

        # Extract k×k patches with stride=1; result has L = H*W locations.
        patches = self.unfold(x_pad)  # (B, C*k*k, H*W)

        # Reshape to (B, C, H, W, k, k) so we can mean over the last two dims (-1, -2).
        patches = patches.view(B, C, k, k, H, W).permute(0, 1, 4, 5, 2, 3)  # (B, C, H, W, k, k)

        # Mean over the k×k window (dims -1 and -2), yielding (B, C, H, W).
        out = patches.mean(dim=(-1, -2))

        return out


def rgb_to_lab(x_rgb: torch.Tensor) -> torch.Tensor:

    x = x_rgb.permute(0, 2, 3, 1)
    mask = x > 0.04045
    x_lin = torch.where(
        mask,
        ((x + 0.055) / 1.055) ** 2.4,
        x / 12.92
    )
    rgb2xyz = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=x_lin.dtype,
        device=x_lin.device
    )
    xyz = torch.einsum("bhwc,cd->bhwd", x_lin, rgb2xyz)

    white_pt = torch.tensor([0.95047, 1.00000, 1.08883],
                            dtype=x_lin.dtype,
                            device=x_lin.device)
    xyz_norm = xyz / white_pt
    eps = 6.0 / 29.0
    threshold = eps ** 3
    f_xyz = torch.where(
        xyz_norm > threshold,
        xyz_norm ** (1.0 / 3.0),
        (xyz_norm / (3 * eps ** 2)) + (4.0 / 29.0)
    )
    L = (116.0 * f_xyz[..., 1]) - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])

    lab = torch.stack([L, a, b], dim=-1)            # (B, H, W, 3)
    lab = lab.permute(0, 3, 1, 2).contiguous()      # (B, 3, H, W)
    return lab

import optuna
from sklearn.metrics import f1_score
import torch.optim as optim

class DepthwiseMean2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        # Depthwise conv: groups=channels
        self.mean_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )
        # Initialize weights to 1/(k*k) and freeze them
        nn.init.constant_(self.mean_conv.weight, 1.0 / (kernel_size * kernel_size))
        self.mean_conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns per-channel mean over each k×k window
        return self.mean_conv(x)

class DepthwiseVariance2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, reduce_channels: bool = False):
        super().__init__()
        padding = kernel_size // 2
        # Same kernel for computing mean of squares
        self.sq_mean_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )
        nn.init.constant_(self.sq_mean_conv.weight, 1.0 / (kernel_size * kernel_size)).to(device)
        self.sq_mean_conv.weight.requires_grad = False
        self.mean_conv = DepthwiseMean2d(channels, self.sq_mean_conv.kernel_size[0])
        self.reduce_channels = reduce_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_of_squares = self.sq_mean_conv(x * x)
        mean_x = self.mean_conv(x)
        var = mean_of_squares - mean_x * mean_x

        if self.reduce_channels:
            var = var.mean(dim=1, keepdim=True)

        return var

class mean_nonzero(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, *), any spatial dims
        B, C = x.shape[:2]

        # Create mask and cast to float -> (B, C, *)
        mask = (x != 0).to(x.dtype)

        # Sum per (B, C)
        total = (x * mask).view(B, C, -1).sum(dim=2)

        # Count per (B, C)
        count = mask.view(B, C, -1).sum(dim=2)

        # Safe mean per-channel
        return total / (count + self.eps)

from torch.func import vmap

class PerImageLABNormalizer(nn.Module):
    def __init__(
        self,
        nbins: int = 100,
        range_l: tuple = (0.0, 100.0),
        range_ab: tuple = (-128.0, 127.0),
        learn_scale: bool = False,
        eps: float = 1e-5
    ):
        super().__init__()
        self.nbins      = nbins
        self.range_l    = range_l
        self.range_ab   = range_ab
        self.eps        = eps

        # Learnable reference peaks: where we want each channel’s mode to land
        mid_l   = 0.5 * (range_l[0] + range_l[1])
        mid_ab  = 0.5 * (range_ab[0] + range_ab[1])
        self.ref_l  = nn.Parameter(torch.rand(1))
        self.ref_a  = nn.Parameter(torch.rand(1))
        self.ref_b  = nn.Parameter(torch.rand(1))

        # Optional learnable scales
        if learn_scale:
            self.scale_l = nn.Parameter(torch.rand(1))
            self.scale_a = nn.Parameter(torch.rand(1))
            self.scale_b = nn.Parameter(torch.rand(1))
        else:
            self.register_buffer('scale_l', torch.rand(1))
            self.register_buffer('scale_a', torch.rand(1))
            self.register_buffer('scale_b', torch.rand(1))

    def forward(self, lab: torch.Tensor) -> torch.Tensor:
        B, C, H, W = lab.shape
        assert C == 3, "Input must be B×3×H×W in LAB space"

        L_chan = lab[:, 0, :, :]
        A_chan = lab[:, 1, :, :]
        B_chan = lab[:, 2, :, :]

        # Compute per‐image peaks
        peak_L = self._batch_second_peak(L_chan, self.range_l)
        peak_A = self._batch_primary_peak(A_chan, self.range_ab)
        peak_B = self._batch_primary_peak(B_chan, self.range_ab)

        # reshape to [B,1,1,1] for broadcasting
        pL = peak_L.view(B, 1, 1, 1)
        pA = peak_A.view(B, 1, 1, 1)
        pB = peak_B.view(B, 1, 1, 1)

        # Normalize each channel: subtract its own peak, add reference, scale
        #Ln = (L_chan.unsqueeze(1) - pL + self.ref_l) * self.scale_l
        Ln = (L_chan.unsqueeze(1) * self.ref_l / pL +100)/200
        #An = (A_chan.unsqueeze(1) - pA + self.ref_a) * self.scale_a
        An = (A_chan.unsqueeze(1) * self.ref_a / pA +128) / 255
        #Bn = (B_chan.unsqueeze(1) - pB + self.ref_b) * self.scale_b
        Bn = (B_chan.unsqueeze(1) * self.ref_b / pB +128) / 255

        return torch.cat([Ln, An, Bn], dim=1) 

    def _batch_second_peak(self, x: torch.Tensor, rng: tuple) -> torch.Tensor:
        B, H, W = x.shape
        device, dtype = x.device, x.dtype
        peaks = torch.zeros(B, device=device, dtype=dtype)

        bin_size = (rng[1] - rng[0]) / self.nbins
        for i in range(B):
            flat = x[i].reshape(-1)
            hist = torch.histc(
                flat, bins=self.nbins, min=rng[0], max=rng[1]
            )
            # top2 modes
            counts, idx = torch.topk(hist, k=2)
            second_bin = idx[1].item()
            peaks[i] = rng[0] + (second_bin + 0.5) * bin_size

        return peaks

    def _batch_primary_peak(self, x: torch.Tensor, rng: tuple) -> torch.Tensor:
        B, H, W = x.shape
        device, dtype = x.device, x.dtype
        peaks = torch.zeros(B, device=device, dtype=dtype)

        bin_size = (rng[1] - rng[0]) / self.nbins
        for i in range(B):
            flat = x[i].reshape(-1)
            hist = torch.histc(
                flat, bins=self.nbins, min=rng[0], max=rng[1]
            )
            max_bin = torch.argmax(hist).item()
            peaks[i] = rng[0] + (max_bin + 0.5) * bin_size

        return peaks

class unify(nn.Module):
    def __init__(self, kernel_size, channels, stride = None, padding = 0):
        super(unify, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.channels = channels
        self.conv = nn.Conv2d(kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, in_channels = channels, out_channels =channels, groups = channels)
        nn.init.constant_(self.conv.weight, 1.0 / (kernel_size * kernel_size))
        self.scale = nn.Parameter(torch.tensor((1), dtype = torch.float32))
    def forward(self, x):
        with autocast(enabled = False):
            x = torch.exp(self.scale * x)
        x = torch.log(self.conv(x)) / self.scale
        return x

class Unify(nn.Module):
    def __init__(self,
                 kernel_size,
                 channels,
                 stride=None,
                 padding=0,
                 init_scale=1.0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.channels = channels
        # learnable sharpness
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x):
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        patches = F.unfold(x,
                           kernel_size=self.kernel_size,
                           stride=self.stride,
                           padding=self.padding)
        patches = patches.view(N, C, kH * kW, -1)
        m = patches.amax(dim=2, keepdim=True).detach()
        with autocast(enabled=False):
            scaled = self.scale * (patches - m)      # (N,C,kH*kW,L)
            exps   = torch.exp(scaled)               # no overflow: ≤ exp(0)=1
            s      = exps.sum(dim=2, keepdim=True)   # (N,C,1,L)
            out_unf = (torch.log(s) + self.scale * m) / self.scale
        out_unf = out_unf.squeeze(2)
        H_out = (H + 2*pH - kH)//sH + 1
        W_out = (W + 2*pW - kW)//sW + 1
        return out_unf.view(N, C, H_out, W_out)

class SoftPool2d(nn.Module):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=1,
                 init_alpha=1.0,
                 learnable=True,
                 positive=True):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        raw = torch.log(torch.exp(torch.tensor(init_alpha)) - 1.0) if positive \
              else torch.tensor(init_alpha)
        self.raw_alpha = nn.Parameter(raw, requires_grad=learnable)
        self.positive = positive

    @property
    def alpha(self):
        if self.positive:
            return F.softplus(self.raw_alpha)
        return self.raw_alpha

    def forward(self, x):
        # x shape: (N, C, H, W)
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        patches = patches.view(N, C, kH * kW, -1).permute(0, 1, 3, 2)
        a = self.alpha.view(1, 1, 1, 1)
        weights = F.softmax(a * patches, dim=-1)
        out = torch.sum(weights * patches, dim=-1)
        H_out = (H + 2 * pH - kH) // sH + 1
        W_out = (W + 2 * pW - kW) // sW + 1
        return out.view(N, C, H_out, W_out)

class MaxCosineNeighborDistance(nn.Module):

    def __init__(self, kernel_size=7, stride=1, padding=3, dilation=1, eps=1e-8):
        super().__init__()
        assert isinstance(kernel_size, int) and kernel_size >= 1, "kernel_size must be a positive int"
        self.k = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size // 2) if padding is None else padding
        self.eps = eps

        # Prepare unfold operator
        self.unfold = nn.Unfold(kernel_size=self.k, dilation=self.dilation,
                                padding=self.padding, stride=self.stride)

        # Precompute neighbor indices (exclude the center position)
        K = self.k * self.k
        center_flat = (self.k // 2) * self.k + (self.k // 2)  # row-major within the window
        idx = torch.arange(K, dtype=torch.long)
        neighbor_idx = torch.cat([idx[:center_flat], idx[center_flat+1:]], dim=0)  # (K-1,)
        self.register_buffer("neighbor_idx", neighbor_idx)
        self.center_flat = int(center_flat)  # Python int for indexing

    def _out_hw(self, H, W):
        H_out = (H + 2*self.padding - self.dilation*(self.k - 1) - 1) // self.stride + 1
        W_out = (W + 2*self.padding - self.dilation*(self.k - 1) - 1) // self.stride + 1
        return H_out, W_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        #assert C == 3, "Expected x with 3 channels (RGB)"
        cols = self.unfold(x)                     # (B, C*K, L) where K=k*k, L=H_out*W_out
        K = self.k * self.k
        L = cols.shape[-1]

        # Reshape to (B, C, K, L)
        patches = cols.view(B, C, K, L)

        # Center pixel vectors: (B, C, L)
        center = patches[:, :, self.center_flat, :]

        # Neighbor pixel vectors: (B, C, K-1, L)
        neighbors = patches.index_select(dim=2, index=self.neighbor_idx)

        # Cosine similarity between center and each neighbor
        # Dot: (B, K-1, L)
        dot = (center.unsqueeze(2) * neighbors).sum(dim=1)

        # Norms: center -> (B, 1, L); neighbors -> (B, 1, K-1, L)
        center_norm = center.norm(dim=1, keepdim=True).clamp_min(self.eps)
        neighbor_norm = neighbors.norm(dim=1, keepdim=True).clamp_min(self.eps)

        # Denominator aligned to (B, K-1, L)
        denom = (center_norm.unsqueeze(2) * neighbor_norm).squeeze(1)

        cos_sim = dot / denom                         # (B, K-1, L)
        cos_dist = 1.0 - cos_sim                      # (B, K-1, L)

        # Max over neighbors and reshape back to map
        #max_dist, _ = cos_dist.max(dim=1, keepdim=True)  # (B, 1, L)
        max_dist = cos_dist.sum(dim =1, keepdim = True) / 8
        H_out, W_out = self._out_hw(H, W)
        return max_dist.view(B, 1, H_out, W_out)

class mask_context(nn.Module):
    def __init__(self,edge_kernel = 3, conv_channel = 64, variance_kernel = 15, hidden_features = 2048, in_features = 256):
        super(mask_context, self).__init__()
        self.edge_kernel = edge_kernel
        self.conv_channel = conv_channel
        self.variance_kernel = variance_kernel
        self.edge_detect = MaxCosineNeighborDistance(kernel_size=3, padding=1)
        self.pool1 = Unify(kernel_size = 3, stride = 1, channels =3)#, in_channels =3, out_channels =3)#, in_channels = 3, out_channels = 3)
        self.pool2 = nn.Conv2d(kernel_size = 3, stride = 1, in_channels = 3, out_channels = 16)
        self.pool3 = nn.Conv2d(kernel_size = 1, in_channels =3, out_channels = 3)
        #self.conv_inneredge = nn.Conv2d(in_channels =32, out_channels =32, kernel_size = 3, padding =1)
        self.conv_inneredge = Unify(kernel_size = 3, padding =1, channels =1)#SoftPool2d(kernel_size =3)#unify(kernel_size = 3, padding =1, channels = 1)#nn.MaxPool2d(kernel_size = 3, padding =1)
        self.std = DepthwiseVariance2d(kernel_size = 5, channels = 3)
        self.slope = nn.Parameter(torch.tensor((3), dtype = torch.float32))
        self.shape1 = nn.Conv2d(in_channels = 16, kernel_size = 3, stride = 1, out_channels = 16)
        self.shape2 = nn.Conv2d(in_channels = 16, kernel_size = 3, stride = 1, out_channels = 16)
        self.shape3 = nn.Conv2d(in_channels = 16, kernel_size = 3, stride = 1, out_channels = 16)
        self.colorconv = nn.Conv2d(kernel_size = 1, in_channels = 3, out_channels =32)
        self.soft = nn.Softmax2d()
        self.lin1 = nn.LazyLinear(256)
        self.lin2 = nn.LazyLinear(32)
        self.lin3 = nn.Linear(32, 4)
        self.convrelu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.05)
        self.mean_nonzero = mean_nonzero()
        self.drop = nn.Dropout(0.2)
        self.localmean = DepthwiseMean2d(kernel_size = 5, channels = 3)
        self.unify1 = Unify(kernel_size = 3, padding =1, channels = 1)#nn.MaxPool2d(kernel_size = 3, padding = 1)
        self.unify2 =Unify(kernel_size = 3, padding =1, channels = 1)
        self.unify3 = Unify(kernel_size = 3, padding =1, channels = 1)
        self.unify4 = Unify(kernel_size = 3, padding =1, channels = 1)
        self.vectors = nn.Parameter(torch.rand(hidden_features, in_features))
        self.beta = nn.Parameter(torch.randn(1, hidden_features))
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(in_features)))
        self.sigmoid = nn.Sigmoid()
        self.sig_factor1 = nn.Parameter(torch.rand(1))
        self.sig_factor2 = nn.Parameter(torch.rand(1))
        #self.ref_color = nn.Parameter(torch.rand((1,3,1)))
        #self.ref_scale = nn.Parameter(torch.rand((1,3,1)))
        self.norm = PerImageLABNormalizer()
    def forward(self, img):
        batch = img.shape[0]
        x = self.norm(img)
        x = self.pool1(x)
        
        x = self.pool3(x)
        #edge = self.relu(x)
        edge = self.edge_detect(x)

        inner_edge = self.conv_inneredge(edge)
        inner_edge = torch.sigmoid(self.slope * inner_edge)
        inner_edge2 = 1-inner_edge
        mean = self.localmean(x)#self.unify1(self.localmean(x))
        std = self.std(x)#self.unify2(self.std(x))
        mean = self.unify3(mean)
        std = self.unify4(std)
        mean1 = mean * inner_edge
        mean2 = mean * inner_edge2
        std1 = -mean1 *self.sig_factor1 / (std + 0.0001)
        std1 = self.sigmoid(std1)
        #std1 = torch.mean(std1, dim = (-1,-2))
        std1m = torch.mean(std1, dim = (-1,-2))
        std1 = torch.reshape(std1, (batch, -1))
        std2 = -mean2 *self.sig_factor2 / (std + 0.0001)
        std2 = self.sigmoid(std2)
        std1 = torch.mean(std2, dim = (-1,-2))
        std2 = torch.reshape(std2, (batch, -1))
        x = self.pool2(img)
        x = self.relu(x)
        x = self.shape1(x)
        #x = self.relu(x)
        x = self.shape2(x)
        #x = self.relu(x)
        x = self.shape3(x)
        x = torch.reshape(x, (batch, -1))
        
        std = torch.cat((std1, std2, x
                        ), dim = -1)
        
        std = self.lin1(std)
        std = self.relu(std)
        std = self.lin2(std)
        std = self.relu(std)
        std = self.lin3(std)
        return std


from sklearn.metrics import precision_recall_curve, auc

def pr_auc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return pr_auc
trainset = image_load(train_ids, PATH)
#img, labels = image_tensor(train_ids, PATH)
#trainset = TensorDataset(img, labels)
testset = image_load(test_ids, PATH)

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from tqdm import tqdm

scaler = GradScaler()
def focal_loss(logits, targets, alpha, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction="none", weight=alpha)
    p_t = torch.exp(-ce)
    loss = (1 - p_t) ** gamma * ce
    return loss.mean()


class_counts = [400_000, 500, 200, 156]
sample_weights = 1. / torch.tensor([class_counts[y] for y in train_y], dtype=torch.double)
rate = 4e-6#trial.suggest_float('rate', 1e-6, 1e-4, log = True)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
trainloader = DataLoader(trainset, batch_size = 32, sampler = sampler, pin_memory = True)
testloader = DataLoader(testset, batch_size = 128, shuffle = False, pin_memory = True)
net = mask_context().to(device)
decay, no_decay = [], []
for name, param in net.named_parameters():
    if not param.requires_grad:
        continue  # skip frozen weights
    # 1‑D params (biases, norm weights, your tiny scalars) → no decay
    if param.ndim == 1 or name.endswith(".bias"):
        no_decay.append(param)
    # explicitly catch any known scalars you want to protect
    elif any(key in name for key in ["slope", "sig_factor", "scale", "beta", "norm"]):
        no_decay.append(param)
    else:
        decay.append(param)


# Build optimizer with separate parameter groups
optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 1e-5},  # your chosen λ
        {"params": no_decay, "weight_decay": 0.0}
    ],
    lr=rate)
epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=rate,
    total_steps=epochs * len(trainloader),
    pct_start=0.1,
    anneal_strategy='cos',
    final_div_factor=30
)
"""
a1 = 0.2#0.3#trial.suggest_float('a1',0.1, 0.9)
a2 = 0.2#0.5#trial.suggest_float('a2',0.1, 0.9)
a3 = 0.2#0.675#trial.suggest_float('a3',0.1, 0.9)
a4 = 0.2#0.675#trial.suggest_float('a4',0.1, 0.9)
alpha = torch.tensor((a1, a2, a3, a4), dtype = torch.float32).to(device)
for epoch in range(5):
    running_loss = 0
    acc_loss =torch.tensor((0)).to(device)
    net.train(True)
    for i, (x, y) in enumerate(trainloader, 0):
        with autocast(enabled=True):
            x = x.to(device)
            x = rgb_to_lab(x)
            y = y.to(device)
            outputs = net(x)
            loss = focal_loss(outputs, y, alpha = alpha#.reshape(-1,1).float()
                          )# / len(trainloader)
        running_loss += loss.detach()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()

        scaler.update()

        optimizer.zero_grad()
        
        if i % 2000 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
            running_loss =0
    net.eval()
    output_list =[]
    label_list =[]
    for i, (x,y) in enumerate(testloader,0):
        with torch.no_grad():
            x = x.to(device)
            x = rgb_to_lab(x)
            outputs = net(x)
            loss = focal_loss(outputs, y.to(device), alpha = alpha)
            outputs = outputs.type(torch.float32)
            outputs = outputs.detach().cpu().numpy()
            output_list.append(outputs.reshape(-1,4))
            y = y.type(torch.float32)
            label_list.append(y.numpy().reshape(-1,1))

    print(loss)
    output_list = np.vstack(output_list)
    label_list = np.vstack(label_list)
    
    label_list[np.argwhere(label_list == 1)] = 0
    label_list[np.argwhere(label_list == 2)] = 1
    label_list[np.argwhere(label_list == 3)] = 1
    label_list = pd.DataFrame(label_list)
    malignant_chance = pd.DataFrame()
    malignant_chance['malignant'] = output_list[:,2] + output_list[:,3]
    #print('roc auc')
    print(roc_auc_score(label_list, malignant_chance['malignant']))
    #print('pr auc')
    print(pr_auc(label_list, malignant_chance['malignant']))

torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, 'augmented.tar')
"""
net = mask_context()
checkpoint = torch.load('augmented.tar', weights_only=True)
net.load_state_dict(checkpoint['model_state_dict'])

resnet = torchvision.models.resnet18(pretrained = True)
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Linear(in_features = 512, out_features = 64)

for param in net.parameters():
    param.requires_grad = False
net.lin3 = nn.Linear(in_features = 32, out_features = 32)


class combined_resnet(nn.Module):
    def __init__(self,fc = 32):
        super(combined_resnet, self).__init__()
        self.texture = net
        self.resnet = resnet
        self.linear1 = nn.Linear(96, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,4)
    def forward(self, x):
        texture = self.texture(x)
        res = self.resnet(x)
        combine = torch.cat([texture, res], dim = 1)
        combine = self.linear1(combine)
        combine = self.relu(combine)
        combine = self.linear2(combine)
        return combine

ensemble = combined_resnet().to(device)
a1 = 0.2#0.3#trial.suggest_float('a1',0.1, 0.9)
a2 = 0.2#0.5#trial.suggest_float('a2',0.1, 0.9)
a3 = 0.2#0.675#trial.suggest_float('a3',0.1, 0.9)
a4 = 0.2#0.675#trial.suggest_float('a4',0.1, 0.9)
alpha = torch.tensor((a1, a2, a3, a4), dtype = torch.float32).to(device)
rate = 0.0001
ens_opt = torch.optim.Adam(
    ensemble.parameters(),
    lr=rate)
epochs = 5
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=rate,
    total_steps=epochs * len(trainloader),
    pct_start=0.1,
    anneal_strategy='cos',
    final_div_factor=30
)

for epoch in range(5):
    running_loss = 0
    acc_loss =torch.tensor((0)).to(device)
    ensemble.train(True)
    for i, (x, y) in enumerate(trainloader, 0):
        with autocast(enabled=True):
            x = x.to(device)
            x = rgb_to_lab(x)
            y = y.to(device)
            outputs = ensemble(x)
            loss = focal_loss(outputs, y, alpha = alpha#.reshape(-1,1).float()
                          )# / len(trainloader)
        running_loss += loss.detach()
        scaler.scale(loss).backward()
        scaler.step(ens_opt)
        scheduler.step()

        scaler.update()

        ens_opt.zero_grad()
        
        if i % 2000 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
            running_loss =0
    ensemble.eval()
    output_list =[]
    label_list =[]
    for i, (x,y) in enumerate(testloader,0):
        with torch.no_grad():
            x = x.to(device)
            x = rgb_to_lab(x)
            outputs = ensemble(x)
            loss = focal_loss(outputs, y.to(device), alpha = alpha)
            outputs = outputs.type(torch.float32)
            outputs = outputs.detach().cpu().numpy()
            output_list.append(outputs.reshape(-1,4))
            y = y.type(torch.float32)
            label_list.append(y.numpy().reshape(-1,1))

    print(loss)
    output_list = np.vstack(output_list)
    label_list = np.vstack(label_list)
    
    label_list[np.argwhere(label_list == 1)] = 0
    label_list[np.argwhere(label_list == 2)] = 1
    label_list[np.argwhere(label_list == 3)] = 1
    label_list = pd.DataFrame(label_list)
    malignant_chance = pd.DataFrame()
    malignant_chance['malignant'] = output_list[:,2] + output_list[:,3]
    #print('roc auc')
    print(roc_auc_score(label_list, malignant_chance['malignant']))
    #print('pr auc')
    print(pr_auc(label_list, malignant_chance['malignant']))


torch.save({
            'epoch': epoch,
            'model_state_dict': ensemble.state_dict(),
            'optimizer_state_dict': ens_opt.state_dict(),
            'loss': loss
            }, 'ensemble_resnet18.tar')