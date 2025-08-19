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

from tqdm import tqdm
PATH = '/mnt/zpool1/zpool1/docker_dir/Documents/isic_skin'
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
print(labels['class'].unique())

train_meta = train_meta.drop(columns =['patient_id', 'target', 'anatom_site_general', 'image_type', 
                                       'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
                                       'tbp_lv_location', 'tbp_lv_location_simple', 'tbp_tile_type'
                                      ])

#drop age and sex because they are not useful, separate id from metadata
train_meta = train_meta.drop(columns =['isic_id', 'sex', 'age_approx'])

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 0.1, cache_size = 2000, class_weight = 'balanced', probability = True)
from sklearn.model_selection import train_test_split as sksplit
target = labels['class']
isic_list = labels['isic_id']
train_x, test_x, train_y, test_y, train_id, test_id = sksplit(train_meta, target, isic_list, test_size = 0.05, stratify = target, shuffle = True)

classifier.fit(train_x, train_y)

from sklearn.metrics import confusion_matrix
predicted = classifier.predict_proba(test_x)
prediction = pd.DataFrame(predicted)
prediction.columns =[0,1,2,3#,4,5
                    ]
print(confusion_matrix(test_y, prediction.idxmax(axis='columns')))

#convert labels
prediction_output = pd.DataFrame(test_y)
prediction_output.columns = ['actual']
prediction = np.array(prediction.idxmax(axis = 'columns'))
prediction[np.argwhere(prediction== 1)] = 0
prediction[np.argwhere(prediction== 2)] = 1
prediction[np.argwhere(prediction== 3)] = 1
prediction[np.argwhere(prediction== 4)] = 0
prediction[np.argwhere(prediction== 5)] = 0


prediction_output['predicted'] = prediction#.idxmax(axis='columns')
print(prediction_output['predicted'])

#prediction_output['actual'] = prediction_output['actual'].map(label_conversion)
#prediction_output['predicted'] = prediction_output['predicted'].map(label_conversion)
print(prediction_output)
print(confusion_matrix(prediction_output['actual'], prediction_output['predicted']))

malignant_chance = pd.DataFrame(np.sum(np.array(predicted)[:, 2:4], axis =1))
#copy from above conversion, to change names
test_y = np.array(test_y)
test_y[np.argwhere(test_y== 1)] = 0
test_y[np.argwhere(test_y== 2)] = 1
test_y[np.argwhere(test_y== 3)] = 1
test_y[np.argwhere(test_y== 4)] = 0
test_y[np.argwhere(test_y== 5)] = 0
test_y = pd.DataFrame(test_y)
test_y.columns = ['truth']
#test_y['truth'] = test_y['truth'].map(label_conversion)
print(test_y)
print('next')
"""
2024 ISIC Challenge primary prize scoring metric

Given a list of binary labels, an associated list of prediction 
scores ranging from [0,1], this function produces, as a single value, 
the partial area under the receiver operating characteristic (pAUC) 
above a given true positive rate (TPR).
https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

(c) 2024 Nicholas R Kurtansky, MSKCC
"""

import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.metrics import roc_curve, auc, roc_auc_score


def score(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float=0.80) -> float:
    '''
    2024 ISIC Challenge metric: pAUC
    
    Given a solution file and submission file, this function returns the
    the partial area under the receiver operating characteristic (pAUC) 
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.
    
    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        solution: ground truth pd.DataFrame of 1s and 0s
        submission: solution dataframe of predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    '''

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(solution.values)-1)
    
    # flip the submissions to their compliments
    v_pred = -1.0*np.asarray(submission.values)

    max_fpr = abs(1-min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

#     # Equivalent code that uses sklearn's roc_auc_score
#     v_gt = abs(np.asarray(solution.values)-1)
#     v_pred = np.array([1.0 - x for x in submission.values])
#     max_fpr = abs(1-min_tpr)
#     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
#     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
#     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
#     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return(partial_auc)

print('partial_auc')
print(score(test_y, malignant_chance))
#0.054 for SVC C=1 rbf and classweight none, no feature engineering
#0.058 rbf, c1 balanced
#0.082 balanced C0.1
#0.046 if balanced C 0.05

#below for implementing image models
import torch
from torch import nn
from torch.data.utils import TensorDataset, DataLoader, Dataset
device = 'cuda:0'
torch.set_default_dtype('bfloat16')
scaler = torch.amp.GradScaler('cuda:0')

train_x = torch.tensor(train_x)
test_x = torch.tensor(test_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

train_ids = pd.DataFrame((trainy, train_id), columns = ['labels', 'ids'])
test_ids = pd.DataFrame((testy, test_id), columns = ['labels', 'ids'])
class img_loader():
    def __init__(self, path):
        self.path = path
    def load(isic_id):
        with h5py.File(path, 'r') as f:
            img = f[isic_id][()]
            img = np.frombuffer(img, np.unit8)
        return img
            
            
class image_load(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = annotations_file
        self.f = h5py.File(img_dir, 'r')

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = f[idx][()]
        image = torch.frombuffer(image, np.unit8)
        label = self.img_labels.loc[self.img_labels['ids' == idx]]['labels']
        return image, label

trainset = image_load(train_ids, path)
testset = image_load(test_ids, path)

trainloader = DataLoader(trainset, batch_size = 32, shuffle = True)
testloader = DataLoader(testset, batch_size = 64, shuffle = False)

#edge detection with mask?
#use color as vector, calculate cosine distance with surrounding points, max pool
#then conv for average to remove noise -> mask
#calculate color heteorgeneity, tone etc
#output multiple category prediction
#calculate probability by softmax of outputs

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
reference from my old code
unfold = nn.Unfold(kernel_size = min(round(dim), 20), stride = 7, padding = 0, dilation = 1)
"""
class MaxCosineNeighborDistance(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=None, dilation=1, eps=1e-8):
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
        assert C == 3, "Expected x with 3 channels (RGB)"
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
        max_dist, _ = cos_dist.max(dim=1, keepdim=True)  # (B, 1, L)
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

loss_fn = nn.NLLLoss()
class mask_context(nn.Module):
    def __init__(self,edge_kernel = 3, conv_channel = 32, variance_kernel = 8):
        super(mask_context, self).__init__()
        self.edge_kernel = edge_kernel
        sekf.conv_channel = conv_channel
        self.variance_kernel = varaince_kernel
        self.edge_detect = MaxCosineNeighborDistance(padding = 2)
        self.conv_inneredge = nn.Conv2d(in_channels =1, out_channels =1, kernel_size = edge_kernel, padding = 2)
        self.color_conv = nn.Conv2d(in_channels =3, oout_channels =conv_channel, kernel_size =1)
        self.std = LocalVarainceUnfold(, kernel = variance_kernel, padding = varaince_kernel -1)
        self.slope = nn.Parameter(torch.rand((1,1)))
        self.lin1 = nn.Linear(32, 128)
        self.lin2 = nn.Linear(128, 16)
        self.lin3 = nn.Linear(16, 4)
        self.relu = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        #tensor shape = batch x 3 x n x n
        edge = self.edge_detect(x)
        inner_edge = self.conv_inneredge(x)
        inner_edge = torch.norm(torch.exp(self.slope * edge), dim = (-1, -2), keepdim = True)
        x = self.color_conv(x)
        std = self.std(x)
        std = std * inner_edge
        std = torch.sum(std, dim = (-1,-2))
        std = torch.reshape(-1,32)
        std = self.lin1(std)
        std = nn.relu(std)
        std = self.lin2(std)
        std = nn.drop(std)
        std = nn.relu(std)
        std = self.lin3(std)
        std = F.log_softmax(std, dim = -1)
        return std

net = mask_context().to(device)
import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(50):    
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        #with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # get the inputs; data is a list of [inputs, labels]
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        loss /= len(trainloader)
        loss.backward()
        if (i+1)%200 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if i % 500 == 499:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
            running_loss = 0.0



    