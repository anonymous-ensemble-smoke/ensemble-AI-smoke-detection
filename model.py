import pickle
import time
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import glob, os
from Loss import DiceLoss, CombinedLoss
from metrics import *

# command example:
# sbatch --export=EXP_NUM=3,SEED=1 --output=logs/exp3_1.log --job-name=exp3_seed1 run_model.script

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_GPUs = torch.cuda.device_count()
print(device, num_GPUs)
num_cores = os.cpu_count()
print(f"CPU cores {num_cores}")

if len(sys.argv) < 2:
    print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)

exp_num = str(sys.argv[1])

# Check if there is another sys argument for the seed
seed = None
if len(sys.argv) > 2:
    seed = int(sys.argv[2])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    print(f'Seed set to {seed}', flush=True)

with open('configs/exp{}.json'.format(exp_num)) as fn:
    hyperparams = json.load(fn)

if 'region' in hyperparams:
    region = hyperparams['region']
    dict_fp = f'geo_split/{region}.pkl'
else:
    region = None
    dict_fp = './deep_learning/dataset_pointers/pseudo/pseudo.pkl' 

with open(dict_fp, 'rb') as handle:
    data_dict = pickle.load(handle)

data_transforms = transforms.Compose([transforms.ToTensor()])

train_set = SmokeDataset(data_dict['train'], data_transforms)
val_set = SmokeDataset(data_dict['val'], data_transforms)
test_set = SmokeDataset(data_dict['test'], data_transforms)

print(f'Dataset splits in {dict_fp}:\n  Training: {len(train_set)}\n  Validation: {len(val_set)}\n  Testing: {len(test_set)}')

def val_model(dataloader, model, loss_fn, dn_weights):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    iou_dict= {'high': {'int': 0, 'union':0}, 'medium': {'int': 0, 'union':0}, 'low': {'int': 0, 'union':0}}
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)

        high_loss = loss_fn(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        med_loss = loss_fn(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        low_loss = loss_fn(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        loss = dn_weights[0]*high_loss + dn_weights[1]*med_loss + dn_weights[2]*low_loss
        #loss = high_loss + med_loss + low_loss
        test_loss = loss.item()
        total_loss += test_loss
        
        iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
    weighted_iou = get_weighted_iou(iou_dict, dn_weights)

    final_loss = total_loss/len(dataloader)
    print("Validation Loss: {}".format(round(final_loss,8)), flush=True)
    return weighted_iou, final_loss

def train_model(train_dataloader, val_dataloader, model, n_epochs, start_epoch, exp_num, 
                best_val_iou, loss_fn, history, dn_weights):
    if history is None:
        history = dict(train_loss=[], val_loss=[], val_iou=[]) 
    
    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
        model.train()
        torch.set_grad_enabled(True)
        #for batch_data, batch_labels in train_dataloader:
        for data in train_dataloader:
            batch_data, batch_labels = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            #print(torch.isnan(batch_data).any())
            optimizer.zero_grad() # zero the parameter gradients
            preds = model(batch_data)
            high_loss = loss_fn(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
            med_loss = loss_fn(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
            low_loss = loss_fn(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
            loss = dn_weights[0]*high_loss + dn_weights[1]*med_loss + dn_weights[2]*low_loss
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            total_loss += train_loss
        epoch_loss = total_loss/len(train_dataloader)
        # print("Training Loss:   {0}".format(round(epoch_loss,8), epoch+1), flush=True)
        val_iou, val_loss = val_model(val_dataloader, model, loss_fn, dn_weights)
        if isinstance(val_iou, torch.Tensor):
            val_iou = val_iou.item()
        history['val_iou'].append(val_iou)
        history['val_loss'].append(val_loss)
        history['train_loss'].append(epoch_loss)
        
        print('Current history at epoch {}'.format(epoch+1), history)

        curr_time = int(time.time())
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': val_iou,
                    'history': history
                    }
            chkpt_pth = './models/{}_exp{}_best_{}.pth'.format(hyperparams['architecture'], exp_num, curr_time)
            if seed is not None:
                chkpt_pth = './models/{}_exp{}_seed{}_best_{}.pth'.format(hyperparams['architecture'], exp_num, seed, curr_time)
            torch.save(checkpoint, chkpt_pth)
            print('SAVING MODEL:\n', chkpt_pth)
        # elif 'save_every_10epochs' in hyperparams and hyperparams['save_every_10epochs'] == "True":
        if (epoch+1) % 10 == 0:
            checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': val_iou,
                    'history': history
                    }
            chkpt_pth = './models/{}_exp{}_{}.pth'.format(hyperparams['architecture'], exp_num, int(time.time()))
            if seed is not None:
                chkpt_pth = './models/{}_exp{}_seed{}_{}.pth'.format(hyperparams['architecture'], exp_num, seed, int(time.time()))
            torch.save(checkpoint, chkpt_pth)
            print('SAVING MODEL:\n', chkpt_pth)

    return model, history

if not "dn_weights" in hyperparams:
    dn_weights = [3,2,1] 
else:
    dn_weights = list(hyperparams["dn_weights"]) 

print('dn_weights')
print(type(dn_weights), dn_weights)

use_ckpt = hyperparams["use_chkpt"]
BATCH_SIZE = int(hyperparams["batch_size"])
encoder_weights = None # hyperparams['encoder_weights'] # if encoder_weights == 'None': #encoder_weights = None
lr = hyperparams['lr']
loss = hyperparams['loss']

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers = num_cores, # num of subprocesses for data loading
        pin_memory=True # automatic memory pinning, enables fast data transfer to CUDA enabled GPUs
        )
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,
        num_workers = num_cores, # num of subprocesses for data loading
        pin_memory=True # automatic memory pinning, enables fast data transfer to CUDA enabled GPUs
        )
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

n_epochs = 2200
start_epoch = 0
model = smp.create_model( # create any model architecture just with parameters, without using its class
        arch=hyperparams['architecture'],
        encoder_name=hyperparams['encoder'],
        encoder_weights=encoder_weights,
        in_channels=3, # model input channels
        classes=3, # model output channels
)

loss_fn = nn.BCEWithLogitsLoss()
if loss == 'DiceLoss':
    loss_fn = DiceLoss()
elif loss == 'CombinedLoss':
    loss_fn = CombinedLoss(dice_weight=1)

optimizer = torch.optim.Adam(list(model.parameters()), lr=lr) 
model = nn.DataParallel(model, device_ids=[i for i in range(num_GPUs)])
model = model.to(device)
best_val_iou = -100000.0
history = None
if use_ckpt == 'True':  # hyperparams['use_chkpt']:   
    ckpts_lst = glob.glob('./models/{}_exp{}_*.pth'.format(hyperparams['architecture'], exp_num))
    if seed is not None:
        ckpts_lst = glob.glob('./models/{}_exp{}_seed{}_*.pth'.format(hyperparams['architecture'], exp_num, seed))
    # if (len(exp_num)>1) and (len(ckpts_lst) == 0): # exp_num variants i.e. 1.1  and there are no ckpts for that variant. 
    #     temp_exp_num = exp_num[0] 
    #     ckpts_lst = glob.glob('./models/{}_exp{}_*.pth'.format(hyperparams['architecture'], temp_exp_num))

    #     # Get the loss fn param for the model being loaded to see if it matches with current loss fn
    #     with open('configs/exp{}.json'.format(temp_exp_num)) as fn:
    #         temp_hyperparams = json.load(fn)
    #     # if temp_hyperparams['loss'] != loss: # different loss fns
    #     #     use_ckpt_best_val_iou = False
    #     #     print('Loss function is different since ckpt, so best_val_iou from ckpt is not being used.')
    if len(ckpts_lst) > 0:
        ckpt_fp = max(ckpts_lst) # get the latest ckpt
        checkpoint=torch.load(ckpt_fp, map_location=device)#'./models/{}_exp{}_*.pth'.format(hyperparams['architecture'], exp_num)) # insert exp_num of ckpt
        state_dict = checkpoint['model_state_dict']
        first_key = next(iter(state_dict))
        if not first_key.startswith('module.'):
            new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict  # 'module.' is already present in keys
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        # Initialize history if not found in checkpoint
        if 'history' in checkpoint and checkpoint['history'] is not None:
            history = checkpoint['history']
            print('History loaded from checkpoint:', history)
            best_val_iou = max(history['val_iou'])
        else:
            print('No history found in the checkpoint. Initializing new history.')
            history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
            best_val_iou = -100000.0
            
        print('Loading from checkpoint at {} (epoch {}, best_val_iou {})'.format(ckpt_fp, start_epoch, best_val_iou))

start = time.time()
model, history = train_model(train_loader, val_loader, model, n_epochs, start_epoch, exp_num, best_val_iou, loss_fn, history, dn_weights)
dt = time.time() - start
print(f"Elapsed time: {dt:.2f} seconds")

# Plotting
def plot_training_history(history, exp_num, save_dir='train_results'):
    import matplotlib.pyplot as plt
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot IoU
    ax2.plot(history['val_iou'], label='Validation IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.legend()
    ax2.grid(True)
    
    # Add stats as text
    best_val_iou = max(history['val_iou'])
    current_val_loss = history['val_loss'][-1]
    stats_text = f'Experiment: {exp_num}\nBest Val IoU: {best_val_iou:.4f}\nFinal Val Loss: {current_val_loss:.4f}'
    fig.text(0.02, 0.02, stats_text, fontsize=10, va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{save_dir}/exp{exp_num}_training_history.png')
    plt.close()

plot_training_history(history, exp_num)
