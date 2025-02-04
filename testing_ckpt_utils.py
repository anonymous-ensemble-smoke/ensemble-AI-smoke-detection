import pickle
from glob import glob
import json
import skimage
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from TestSmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from metrics import *
import matplotlib.pyplot as plt
import skimage
from datetime import datetime
# from ensemble_utils import *
import pyproj
import cartopy.crs as ccrs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_GPUs = torch.cuda.device_count()
print(device, num_GPUs)
num_cores = os.cpu_count()
print(f"CPU cores {num_cores}")


def get_datetime(fn):
    start = fn.split('/')[-1].split('_')[1][1:-1]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    start_readable = start_dt.strftime('%Y/%m/%d %H:%M UTC')
    #start_readable = start_dt.strftime('%H:%M UTC')
    return start_readable

def get_mesh(num_pixels):
    x = np.linspace(0,num_pixels-1,num_pixels)
    y = np.linspace(0,num_pixels-1,num_pixels)
    X, Y = np.meshgrid(x,y)
    return X,Y

def coords_from_fn(fn, res=1000, img_size=256): # img_size - number of pixels
    lat, lon = get_center_lat_lon(fn)
    lcc_proj = pyproj.Proj(get_proj())
    x, y = lcc_proj(lon,lat)
    dist = int(img_size/2*res)
    lon_0, lat_0 = lcc_proj(x-dist, y-dist, inverse=True) # lower left
    lon_1, lat_1 = lcc_proj(x+dist, y+dist, inverse=True) # upper right
    lats = np.linspace(lat_1, lat_0, 5)
    lons = np.linspace(lon_0, lon_1, 5)
    return lats, lons

def get_center_lat_lon(fn):
    fn_split = fn.split('.tif')[0].split('_')
    lat = fn_split[-3]
    lon = fn_split[-2]
    return lat, lon

def get_proj():
    lcc_proj = ccrs.LambertConformal(central_longitude=262.5,
                                     central_latitude=38.5,
                                     standard_parallels=(38.5, 38.5),
                                     globe=ccrs.Globe(semimajor_axis=6371229,
                                                      semiminor_axis=6371229))

    return lcc_proj

def compute_overall_iou(preds, truths):
    densities = ['heavy', 'medium', 'low']
    intersection = 0
    union = 0
    for idx, level in enumerate(densities):
        pred = preds[:,idx,:,:]
        true = truths[:,idx,:,:]
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        intersection += (pred + true == 2).sum()
        union += (pred + true >= 1).sum()
    try:
        iou = intersection / union
        return iou
    except Exception as e:
        print(e)
    return 0

def get_data_dict_from_fn(truth_fn):
    data_fn = truth_fn.replace('truth','data')
    data_dict = {'find': {'truth': [truth_fn], 'data': [data_fn]}}
    return data_dict

def get_pred(dataloader, model, device):
    model.eval()
    torch.set_grad_enabled(False)
    for idx, data in enumerate(dataloader):
        batch_data, batch_labels, truth_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        out = model(batch_data)
        iou = compute_overall_iou(out, batch_labels)
        # iou = iou.cpu().detach().numpy()
        # print(np.round(iou, 4))
        pred = torch.sigmoid(out)
        pred = (pred > 0.5) * 1
        pred = pred.squeeze(0).cpu().detach().numpy()
    return pred, iou

def run_model_single_fn(fn, chkpt_path = './models/ckpt2.pth', model=''):
    data_dict = get_data_dict_from_fn(fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([transforms.ToTensor()])
    test_set = SmokeDataset(data_dict['find'], data_transforms)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    if model == '':
        model = smp.DeepLabV3Plus(
                encoder_name="timm-efficientnet-b2",
                encoder_weights=None, # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3, # model input channels
                classes=3, # model output channels
        )
        model = model.to(device)
        
        checkpoint = torch.load(chkpt_path, map_location=torch.device(device), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    pred, iou = get_pred(test_loader, model, device)
    return pred, iou

def ens_probabilities(multimodel_preds):
    # Convert predictions to float type for averaging
    multimodel_preds = [pred.float() if isinstance(pred, torch.Tensor) else torch.tensor(pred, dtype=torch.float).to(device) for pred in multimodel_preds]
    
    # Initialize result tensor with float type
    result = torch.zeros_like(multimodel_preds[0], dtype=torch.float).to(device)
    
    # Average the predictions
    for pred in multimodel_preds:
        result += torch.sigmoid(pred)
    result = result / len(multimodel_preds)

    # Convert to binary predictions
    result = (result > 0.5)* 1 #.float()
    
    return result

def ens_probabilities_no_sigmoid(multimodel_preds):
    # Convert predictions to float type for averaging
    multimodel_preds = [pred.float() if isinstance(pred, torch.Tensor) else torch.tensor(pred, dtype=torch.float).to(device) for pred in multimodel_preds]
    
    # Initialize result tensor with float type
    result = torch.zeros_like(multimodel_preds[0], dtype=torch.float).to(device)
    
    # Average the predictions
    for pred in multimodel_preds:
        result += pred
    result = result / len(multimodel_preds)

    # Convert to binary predictions
    result = (result > 0.5)* 1 #.float()
    
    return result

def plot_test_results_smokeviz(pred, fn, save_fig = False):
    preds = np.dstack([pred[0], pred[1], pred[2]])
    data_fn = fn.replace('truth', 'data')
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    truths = skimage.io.imread(fn, plugin='tifffile')
    lat, lon = coords_from_fn(fn)
    num_pixels = RGB.shape[1]
    X, Y = get_mesh(num_pixels)
    colors = ['red', 'orange', 'yellow']
    fig, ax = plt.subplots(1, 2, figsize=(16,8))

    ax[0].imshow(RGB)
    ax[1].imshow(RGB)
    for idx in reversed(range(3)):
        ax[0].contour(X,Y,truths[:,:,idx],levels =[.99],colors=[colors[idx]])
        ax[1].contour(X,Y,preds[:,:,idx],levels =[.99],colors=[colors[idx]])

    ax[0].set_yticks(np.linspace(0,255,5), np.round(lat,2), fontsize=12)
    ax[0].set_ylabel('latitude (degrees)', fontsize=16)
    ax[0].set_xticks(np.linspace(0,255,5), np.round(lon,2), fontsize=12)
    ax[0].set_xlabel('longitude (degrees)', fontsize=16)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[0].set_title('analyst annotation',fontsize=18)
    ax[1].set_title('model prediction',fontsize=18)
    plt.suptitle(get_datetime_from_fn(fn), fontsize=18)

    #plt.tight_layout(pad=0)#, h_pad=-.5)
    plt.subplots_adjust(wspace=0)
    if save_fig:
        results_dir = './paper_figures/scripts/results/'
        fn_head = fn.split('/')[-1].split('.tif')[0]
        plt.savefig('{}{}_results.png'.format(results_dir, fn_head), dpi=300)
    plt.show()

def test_multi_model(dataloader, model_list, BCE_loss, get_individual_ious=True):
    print('Ensembling {} models'.format(len(model_list)))
    # ens_total_loss = 0.0
    ens_iou_dict= {'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}}
    individual_model_iou_dicts = []
    for model_i in range(len(model_list)):
        individual_model_iou_dicts.append({'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}})

    for idx, data in enumerate(dataloader):
        preds_list = []
        batch_data, batch_labels, truth_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        for model_i, model in enumerate(model_list):
            model.eval()
            torch.set_grad_enabled(False)
            preds = model(batch_data).to(device, dtype=torch.float)
            preds_list.append(preds)
            if get_individual_ious:
                curr_iou_dict = individual_model_iou_dicts[model_i]
                curr_iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', curr_iou_dict)
                curr_iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', curr_iou_dict)
                curr_iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', curr_iou_dict)
                individual_model_iou_dicts[model_i] = curr_iou_dict
        # Peform ensembling
        ens_pred = ens_probabilities(preds_list).to(device, dtype=torch.float)
        # high_loss = BCE_loss(ens_pred[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        # med_loss = BCE_loss(ens_pred[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        # low_loss = BCE_loss(ens_pred[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        # loss = 3*high_loss + 2*med_loss + low_loss
        # test_loss = loss.item()
        # ens_total_loss += test_loss
    
        ens_iou_dict= compute_iou(ens_pred[:,0,:,:], batch_labels[:,0,:,:], 'high', ens_iou_dict)
        ens_iou_dict= compute_iou(ens_pred[:,1,:,:], batch_labels[:,1,:,:], 'medium', ens_iou_dict)
        ens_iou_dict= compute_iou(ens_pred[:,2,:,:], batch_labels[:,2,:,:], 'low', ens_iou_dict)

    iou_list = get_iou_by_density(ens_iou_dict)
    individual_iou_lists = []
    for model_i in range(len(model_list)):
        individual_iou_lists.append(get_iou_by_density(individual_model_iou_dicts[model_i]))
    final_loss = 0 # ns_total_loss/len(dataloader)
        
    return iou_list, final_loss, individual_iou_lists

def save_test_results(truth_fn, preds, dir_num, iou_dict, ckpt_fn, sample_ds=False):
    results_str = '_sample' if sample_ds else ''
    save_loc = os.path.join(os.getcwd(), 'test_results{}/{}/{}/'.format(results_str, ckpt_fn, dir_num))
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    
    truth_fn = truth_fn[0]
    data_fn = truth_fn.replace('truth', 'data')
    coords_fn = truth_fn.replace('truth', 'coords')
        
    skimage.io.imsave(save_loc + 'preds.tif', preds)
    [high_iou, med_iou, low_iou, overall_iou] = get_iou_by_density(iou_dict)

    fn_info = {'data_fn': data_fn,
               'truth_fn': truth_fn,
               'coords_fn': coords_fn,
               }

    try:
        fn_info['low_iou'] = str(low_iou.cpu().numpy())
    except AttributeError:
        fn_info['low_iou'] = str(low_iou)
    try:
        fn_info['medium_iou'] = str(med_iou.cpu().numpy())
    except AttributeError:
        fn_info['medium_iou'] = str(med_iou)
    try:
        fn_info['high_iou'] = str(high_iou.cpu().numpy())
    except AttributeError:
        fn_info['high_iou'] = str(high_iou)
    try:
        fn_info['overall_iou'] = str(overall_iou.cpu().numpy())
    except AttributeError:
        fn_info['overall_iou'] = str(overall_iou)

    json_object = json.dumps(fn_info, indent=4)
    with open(save_loc + "fn_info.json", "w") as outfile:
        outfile.write(json_object)

def get_test_results(ckpt_fn, dir_num, sample_ds=False):
    results_str = '_sample' if sample_ds else ''
    save_loc = os.path.join(os.getcwd(), 'test_results{}/{}/{}/'.format(results_str, ckpt_fn, dir_num)) 
    preds_fp = os.path.join(save_loc, 'preds.tif')
    preds = skimage.io.imread(preds_fp, plugin='tifffile') 
    fn_info_fp = os.path.join(save_loc, 'fn_info.json')
    with open(fn_info_fp) as fn:
        fn_info = json.load(fn)
    return preds, fn_info

def load_ckpt(exp_num, base_model=False, print_history=False, seed=None):    
    with open('configs/exp{}.json'.format(exp_num)) as fn:
        hyperparams = json.load(fn)
    use_ckpt = hyperparams['use_chkpt']
    encoder_weights = None
    model = smp.create_model(
            arch=hyperparams['architecture'],
            encoder_name=hyperparams['encoder'],
            encoder_weights=encoder_weights,
            in_channels=3,          
            classes=3,
    )
    lr = hyperparams['lr']
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr) 
    model = nn.DataParallel(model, device_ids=[i for i in range(num_GPUs)])
    model = model.to(device)
    best_val_iou = -100000.0
    ckpt_fp, checkpoint = None, None

    if use_ckpt:
        if base_model:
            # Load base model
            ckpt_fp = "./deep_learning/models/DeepLabV3Plus_exp0_1731375075.pth"
            # ckpt_fp = './models/DeepLabV3Plus_exp1_1726250084.pth'
        else:
            use_best_model = ''
            if 'use_best_model' in hyperparams:
                if hyperparams['use_best_model'] == "True":
                    use_best_model = '_best'

            # print('use_best_model:', use_best_model)

            ckpts_lst = glob('./models/{}_exp{}_*.pth'.format(hyperparams['architecture'], exp_num))
            if seed is not None:
                ckpts_lst = [ckpt for ckpt in ckpts_lst if (f'_seed{seed}_' in ckpt)]

            if use_best_model == '':
                ckpts_lst = [ckpt for ckpt in ckpts_lst if '_best_' not in ckpt]
            
            if (len(exp_num)>1) and (len(ckpts_lst) == 0): # if no ckpt for exp_num, try to load ckpt for base exp_num
                temp_exp_num = exp_num[0]
                ckpts_lst = glob('./models/{}_exp{}_*.pth'.format(hyperparams['architecture'], temp_exp_num))
            
            if len(ckpts_lst) > 0:
                ckpt_fp = max(ckpts_lst)
            else: 
                raise FileNotFoundError('no ckpt for exp_num {}'.format(exp_num))
            
        checkpoint=torch.load(ckpt_fp, map_location=torch.device(device))#'./models/{}_exp{}_*.pth'.format(hyperparams['architecture'], exp_num)) # insert exp_num of ckpt
        state_dict = checkpoint['model_state_dict']
        first_key = next(iter(state_dict))
        if not first_key.startswith('module.'):
            new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict  # 'module.' is already present in keys
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'history' in checkpoint:
            history = checkpoint['history']
            if print_history: print('History loaded from checkpoint:', history)

        if 'val_iou' in checkpoint:
            best_val_iou = checkpoint['val_iou']
        print('Loading from checkpoint at {} (epoch {}, best_val_iou {})'.format(ckpt_fp, start_epoch, best_val_iou))
#        best_loss = checkpoint['loss']
        # print('ckpt {}\n best_val_iou: {} \nepoch: {}'.format(ckpt_fp, best_val_iou, start_epoch))
    return ckpt_fp, checkpoint, model, optimizer

def test_model(dataloader, model, BCE_loss, ckpt_fp, exp_num, sample_ds=False):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    iou_dict= {'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 
               'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 
               'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}}

    with open('configs/exp{}.json'.format(exp_num)) as fn:
        hyperparams = json.load(fn)
    dn_weights = list(hyperparams.get("dn_weights", [3,2,1]))  # Default to [3,2,1] if not specified

    for idx, data in enumerate(dataloader):
        batch_data, batch_labels, truth_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)

        high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        loss = dn_weights[0]*high_loss + dn_weights[1]*med_loss + dn_weights[2]*low_loss  # Updated weighted loss
        test_loss = loss.item()
        total_loss += test_loss
        iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
        # if idx < max_num:
        #     # print(idx)
        #     save_test_results(truth_fn, preds.detach().to('cpu').numpy(), idx, iou_dict, os.path.split(ckpt_fp)[1], sample_ds=sample_ds)
        # print(preds.detach().to('cpu').numpy().shape) ### (1, 3, 256, 256)
        #    break
    
    [high_iou, med_iou, low_iou, iou] = get_iou_by_density(iou_dict)

    final_loss = total_loss/len(dataloader)
    # print("Testing Loss: {}\n".format(round(final_loss,8)), flush=True)
    return final_loss, iou_dict, [high_iou, med_iou, low_iou, iou]

# def get_lat_lon(fn, data_loc="./sample_data/"):
#     coords_fn = glob(data_loc + "coords/*/*/" + fn)[0]
#     lat_lon = skimage.io.imread(coords_fn, plugin='tifffile')
#     lat = lat_lon[:,:,0]
#     lon = lat_lon[:,:,1]
#     return lat[::63,0], lon[-1,::63]

def get_data(fn, data_loc="./sample_data/"):
    data_fn = glob(data_loc + "data/*/*/" + fn)[0]
    truth_fn = glob(data_loc + "truth/*/*/" + fn)[0]
    RGB = skimage.io.imread(data_fn, plugin='tifffile')
    truths = skimage.io.imread(truth_fn, plugin='tifffile')
    lat, lon = get_center_lat_lon(fn)
    return RGB, truths, lat, lon

def get_datetime_from_fn(fn):
    start = fn.split('_')[1][1:-1]
    start_dt = datetime.strptime(start, '%Y%j%H%M%S')
    start_readable = start_dt.strftime('%Y/%m/%d %H:%M UTC')
    return start_readable

def plot_exp_test_results(exp_num_list, test_fn, test_idx, 
                          save=False, with_ensemble=False, 
                          ensemble_preds=torch.tensor([]), ens_iou=0,
                         add_lat_lon=False, sample_ds=False):
    data_loc = './data/sample_data/'
    RGB, truths, lat, lon = get_data(test_fn, data_loc=data_loc)
    # print('RGB.shape, truths.shape, lat.shape, lon.shape')
    # print(RGB.shape, truths.shape, lat.shape, lon.shape)
    num_pixels = RGB.shape[1]
    
    # Determine plot layout
    num_panels = len(exp_num_list) + 1  # Truth + models
    if with_ensemble:
        num_panels += 1  # Add ensemble panel
    
    figwidth = 20 if (num_panels > 2) else 10
    nrows = 1
    ncols = num_panels
    
    # Create figure
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, 
                          figsize=(figwidth, 5), dpi=100)
    ax = ax.ravel()
    
    # Plot ground truth
    ax[0].imshow(RGB)
    X, Y = get_mesh(num_pixels)
    colors = ['red', 'orange', 'yellow']
    labels = ['high', 'medium', 'light']
    for idx in reversed(range(3)):
        ax[0].contour(X, Y, truths[:,:,idx], levels=[.99], colors=[colors[idx]])
    ax[0].set_title('HMS Analyst Annotation', fontsize=12)
    
    # Plot individual model predictions
    for exp_num_idx, exp_num in enumerate(exp_num_list):
        # Load model and predictions
        ckpt_fp, checkpoint, model, optimizer = load_ckpt(exp_num)
        ckpt_fn = os.path.split(ckpt_fp)[1]
        preds, fn_info = get_test_results(ckpt_fn, test_idx, sample_ds=sample_ds)
        print('preds.shape',preds.shape)
        iou = fn_info['overall_iou']
        
        # Plot RGB image
        ax[exp_num_idx + 1].imshow(RGB)
        
        # Plot model predictions
        preds = torch.tensor(preds)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5) * 1
        for idx in reversed(range(3)):
            ax[exp_num_idx + 1].contour(X, Y, preds.squeeze()[:,:,idx], 
                                      levels=[.99], colors=[colors[idx]])
        
        # Add title and IoU score
        arch_name = ckpt_fn.split('_')[0]
        ax[exp_num_idx + 1].set_title(f'{arch_name}', fontsize=12)
        ax[exp_num_idx + 1].set_xlabel(f'IoU {float(iou):.4f}', fontsize=15)
        ax[exp_num_idx + 1].set_xticks([])
        ax[exp_num_idx + 1].set_yticks([])
    
    # Plot ensemble predictions if provided
    if with_ensemble:
        ax[-1].imshow(RGB)
        for idx in reversed(range(3)):
            ax[-1].contour(X, Y, ensemble_preds.squeeze()[:,:,idx],
                          levels=[.99], colors=[colors[idx]])
        ax[-1].set_xlabel(f'IoU {float(ens_iou):.4f}', fontsize=15)
        ax[-1].set_title('Ensemble', fontsize=12)
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
    
    if add_lat_lon:
        ax[0].set_yticks(np.linspace(0, 255, 5), np.round(lat, 2))
        ax[0].set_xticks(np.linspace(0, 255, 5), np.round(lon, 2))
        ax[0].set_ylabel('longitude', fontsize=12)
        ax[0].set_xlabel('latitude', fontsize=12)
    
    # fig.supylabel('latitude (degrees)', fontsize=16)
    # fig.supxlabel('longitude (degrees)', fontsize=16)
    fig.suptitle(get_datetime_from_fn(test_fn), fontsize=18)
    plt.tight_layout(pad=2)
    if save:
        # Create results directory
        results_dir = 'test_results'
        if sample_ds:
            results_dir += '_sample'
        os.makedirs(results_dir, exist_ok=True)
        
        # Create descriptive filename
        model_str = '_'.join([f'exp{exp}' for exp in exp_num_list])
        if with_ensemble:
            model_str += '_with_ensemble'
        
        timestamp = test_fn.split('_')[1]  # Extract timestamp from test filename
        filename = f'{results_dir}/comparison_{model_str}_{timestamp}_idx{test_idx}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def get_param_from_config(exp_num, param_name):
    with open('configs/exp{}.json'.format(exp_num)) as fn:
        hyperparams = json.load(fn)
    
    # Handle special cases and formatting
    if param_name == 'loss':
        return hyperparams.get(param_name, 'BCEWithLogitsLoss')
    elif param_name == 'architecture':
        return hyperparams.get(param_name, 'Unknown')
    elif param_name == 'encoder':
        return hyperparams.get(param_name, 'Unknown')
    elif param_name == 'lr':
        return f"{hyperparams.get(param_name, 0.0):.1e}"
    elif param_name == 'batch_size':
        return str(hyperparams.get(param_name, 'Unknown'))
    else:
        return str(hyperparams.get(param_name, 'Unknown'))


# def plot_densities_from_processed_data(fn, data_loc="./sample_data/", close=False, save=False):
#     RGB, truths, lat, lon = get_data(fn, data_loc)
#     num_pixels = RGB.shape[1]
#     X, Y = get_mesh(num_pixels)
#     colors = ['red', 'orange', 'yellow']
#     plt.figure(figsize=(8, 6),dpi=100)

#     plt.imshow(RGB)
#     for idx in reversed(range(3)):
#         plt.contour(X,Y,truths[:,:,idx],levels =[.99],colors=[colors[idx]])
#     plt.tight_layout(pad=0)
#     plt.yticks(np.linspace(0,255,5), np.round(lat,2), fontsize=12)
#     plt.ylabel('latitude (degrees)', fontsize=16)
#     plt.xticks(np.linspace(0,255,5), np.round(lon,2), fontsize=12)
#     plt.xlabel('longitude (degrees)', fontsize=16)
#     plt.title(get_datetime_from_fn(fn), fontsize=18)
#     plt.tight_layout(pad=0)#, h_pad=-.5)
#     if save:
#         plt.savefig('densities.png', dpi=300)
#     plt.show()
#     if close:
#         plt.close()

# def plot_RGB(fn, data_loc="./sample_data/"):
#     data_fn = glob(data_loc + "data/*/*/" + fn)[0]
#     print(get_datetime_from_fn(fn))
#     RGB = skimage.io.imread(data_fn, plugin='tifffile')
#     lat, lon = get_lat_lon(fn, data_loc)
#     plt.figure(figsize=(8, 6),dpi=100)
#     plt.imshow(RGB)
#     plt.yticks(np.linspace(0,255,5), np.round(lat,2), fontsize=12)
#     plt.ylabel('latitude (degrees)', fontsize=16)
#     plt.xticks(np.linspace(0,255,5), np.round(lon,2), fontsize=12)
#     plt.xlabel('longitude (degrees)', fontsize=16)
#     plt.title('RGB',fontsize=24)
#     plt.tight_layout(pad=0)
#     plt.show()

# def plot_R_G_B_RGB(fn, data_loc="./sample_data/"):
#     data_fn = glob(data_loc + "data/*/*/" + fn)[0]
#     coords_fn = glob(data_loc + "coords/*/*/" + fn)[0]

#     fig, ax = plt.subplots(1, 4, figsize=(40,15))
#     RGB = skimage.io.imread(data_fn, plugin='tifffile')
#     coords = skimage.io.imread(coords_fn, plugin='tifffile')
#     print(fn)
#     print(np.round(coords[128][128][0],2), ',', np.round(coords[128][128][1],2))
#     labels = ['Red', 'Green', 'Blue', 'RGB']
#     cmaps = ['Reds', 'Greens', 'Blues']
#     for idx in range(4):
#         if idx < 3:
#             #ax[idx].imshow(RGB[:,:,idx], cmap='Greys_r')
#             ax[idx].imshow(RGB[:,:,idx], cmap=cmaps[idx])
#         else:
#             ax[idx].imshow(RGB)
#         ax[idx].set_yticks([])
#         ax[idx].set_xticks([])
#         ax[idx].set_title(labels[idx],fontsize=30)
#     #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = .02)
#     plt.tight_layout(pad = 2)
#     #plt.margins(0,0)
#     plt.show()

# def plot_R_G_B(fn, data_loc="./sample_data/"):
#     data_fn = glob(data_loc + "data/*/*/" + fn)[0]
#     coords_fn = glob(data_loc + "coords/*/*/" + fn)[0]
#     fig, ax = plt.subplots(1, 3, figsize=(15,30))
#     RGB = skimage.io.imread(data_fn, plugin='tifffile')
#     coords = skimage.io.imread(coords_fn, plugin='tifffile')
#     print(fn)
#     print("center lat, lon: ({}, {})".format(np.round(coords[128][128][0],2), np.round(coords[128][128][1],2)))
#     labels = ['Red Channel', '\"Green\" Channel', 'Blue Channel']
#     for idx in range(3):
#         ax[idx].imshow(RGB[:,:,idx], cmap='Greys_r')
#         ax[idx].set_yticks([])
#         ax[idx].set_xticks([])
#         ax[idx].set_title(labels[idx],fontsize=20)
#     plt.tight_layout(pad=1)
#     plt.show()

# def plot_labels(fn, data_loc="./sample_data/"):
#     truth_fn = glob(data_loc + "truth/*/*/" + fn)[0]
#     #coords_fn = glob(data_loc + "coords/*/*/" + fn)[0]
#     fig, ax = plt.subplots(1, 3, figsize=(15,30))
#     truths = skimage.io.imread(truth_fn, plugin='tifffile')
#     #coords = skimage.io.imread(coords_fn, plugin='tifffile')
#     print(fn)
#     #print(np.round(coords[128][128][0],2), ',', np.round(coords[128][128][1],2))
#     labels = ['high', 'medium', 'light']
#     for den in range(3):
#         ax[den].imshow(truths[:,:,den], cmap='Greys_r', vmin=0, vmax=1)
#         ax[den].set_yticks([])
#         ax[den].set_xticks([])
#         ax[den].set_title(labels[den], fontsize=20)
#     plt.tight_layout(pad=1)
#     plt.show()


# def plot_True_Color(fn, data_loc="./sample_data/"):
#     data_fn = glob(data_loc + "data/*/*/" + fn)[0]
#     #print(get_datetime_from_fn(fn))
#     RGB = skimage.io.imread(data_fn, plugin='tifffile')
#     lat, lon = get_lat_lon(fn, data_loc)
#     plt.figure(figsize=(8, 6),dpi=100)
#     plt.imshow(RGB)
#     plt.yticks(np.linspace(0,255,5), np.round(lat,2), fontsize=12)
#     plt.ylabel('latitude (degrees)', fontsize=16)
#     plt.xticks(np.linspace(0,255,5), np.round(lon,2), fontsize=12)
#     plt.xlabel('longitude (degrees)', fontsize=16)
#     plt.title('True Color \n {}'.format(get_datetime_from_fn(fn)),fontsize=18)
#     plt.tight_layout(pad=0)
#     plt.show()
