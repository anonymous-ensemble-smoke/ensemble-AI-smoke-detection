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
from tqdm import tqdm
from testing_ckpt_utils import save_test_results, get_test_results, load_ckpt, test_model, ens_probabilities, test_multi_model, get_param_from_config
from tabulate import tabulate
from Loss import DiceLoss, CombinedLoss, get_loss_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_GPUs = torch.cuda.device_count()
print(device, num_GPUs)
num_cores = os.cpu_count()
print(f"CPU cores {num_cores}")

if len(sys.argv) < 2:
    print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)

params_to_show_list = []
# Add parameter selection for results table
if len(sys.argv) < 3:
    param_to_show = 'architecture'  # default to showing architecture type
    print('\n No parameter specified for results table. Defaulting to architecture type.', flush=True)
else:
    for param in sys.argv[2:]:
        params_to_show_list.append(param)

# Input format #
# 1_1.2.1_3_T
# => test exp 1, 1.2.1, 3 individually; and ensemble them together
# 1_F
# => test exp 1; don't ensemble
# 0_1.0.2_T
# => test base model and exp 1.0.2 indivdually; and ensemble them togethwer
# if ensemble = 'S' or 'ST', then ensemble across different seeds
input = sys.argv[1]
input_list = str(input).split('_')
ensemble = input_list[-1]
input_list.remove(ensemble)
# print(input, input_list, ensemble)

dict_fp = './deep_learning/dataset_pointers/pseudo/pseudo.pkl'
with open(dict_fp, 'rb') as handle:
    data_dict = pickle.load(handle)

data_transforms = transforms.Compose([transforms.ToTensor()])
test_set = SmokeDataset(data_dict['test'], data_transforms)

print('there are {} testing samples in {}'.format(len(test_set), dict_fp))

BCE_loss = nn.BCEWithLogitsLoss()
BATCH_SIZE = 128
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    drop_last=True,
    num_workers=num_cores,
    pin_memory=True
)

results_table = ["Experiment",'Epoch',]
for param in params_to_show_list:
    results_table.append(param)
if 'S' in ensemble:
    results_table.append('Seed')
results_table += ["High IoU",  "Medium IoU", "Low IoU",  "Overall IoU"]
results_table = [results_table]
model_list = []

for exp_num in tqdm(input_list):
    if exp_num[0:4] == '1.6.':
        region = get_param_from_config(exp_num, 'region')
        print('Region: ', region)
        dict_fp = f'geo_split/{region}.pkl'
        print(dict_fp)
        with open(dict_fp, 'rb') as handle:
            data_dict = pickle.load(handle)
        test_set = SmokeDataset(data_dict['test'], data_transforms)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            drop_last=True,
        )

    # print(f"\nTesting experiment {exp_num}")
    loss_fn = get_loss_function(exp_num)
    if 'S' in ensemble:
        if exp_num in ['3','1']:
            for seed in range(1,13):
                ckpt_fp, checkpoint, model, optimizer = load_ckpt(exp_num, seed=seed)
                model_list.append(model)
                name = os.path.split(ckpt_fp)[1]
                # final_loss, iou_dict, iou_list = test_model(test_loader, model, loss_fn, ckpt_fp, exp_num)
                row = [exp_num, checkpoint['epoch']]
                for param in params_to_show_list:
                    param_value = get_param_from_config(exp_num, param)
                    row.append(param_value)
                row.append(seed)
                row += [0,0,0,0] #blank iou list #iou_list
                results_table.append(row)
        else:
            ckpt_fp, checkpoint, model, optimizer = load_ckpt(exp_num)
            model_list.append(model)
            name = os.path.split(ckpt_fp)[1]
            # final_loss, iou_dict, iou_list = test_model(test_loader, model, loss_fn, ckpt_fp, exp_num)
            row = [exp_num, checkpoint['epoch']]
            for param in params_to_show_list:
                param_value = get_param_from_config(exp_num, param)
                row.append(param_value)
            row.append('--')
            row += [0,0,0,0] # blank iou list # iou_list
            results_table.append(row)
    else: 
        ckpt_fp, checkpoint, model, optimizer = load_ckpt(exp_num)
        model_list.append(model)
        arch_name = os.path.split(ckpt_fp)[1].split('_')[1]
        # final_loss, iou_dict, iou_list = test_model(test_loader, model, loss_fn, ckpt_fp, exp_num)
        row = [exp_num, checkpoint['epoch']]
        for param in params_to_show_list:
            param_value = get_param_from_config(exp_num, param)
            row.append(param_value)
        row += [0,0,0,0] # blank iou list # iou_list
        results_table.append(row)

# print(tabulate(results_table,headers="firstrow", 6tablefmt='grid'))

### Testing an ensemble of models
incremental_ensemble_size = True
incremental_results = {
    # 'ascending': { # add models in order of input_list
    #     'iou_list': [], # will be a list of lists of iou values for each density level
    # },
    'descending': { # add models in reverse order of input_list
        'iou_list': [],
    }
} 
if ensemble == 'T' or 'S' in ensemble:
    print('\n\n================== MULTI-MODEL RESULTS ==================')
    print('exp num list {}'.format(input_list))
    
    loss_fn = nn.BCEWithLogitsLoss()  # Use BCE for ensemble testing    
    if not input_list[0][0:4] == '1.6.':
        iou_list_cpu=[]
        if incremental_ensemble_size:
            print('Incremental ensemble size')
            for i in range(1, len(model_list)+1):
                iou_list, final_loss, individual_iou_lists = test_multi_model(test_loader, model_list[:i], loss_fn)
                iou_list_cpu = [iou.cpu() for iou in iou_list]
                incremental_results['descending']['iou_list'].append(iou_list_cpu)
                row = ['Ensemble ', 'N = {}'.format(i+1)] 
                for param in params_to_show_list:
                    row.append('--')
                if 'S' in ensemble: 
                    row.append('--')
                row += iou_list_cpu
                results_table.append(row)
        else:
            iou_list, final_loss, individual_iou_lists = test_multi_model(test_loader, model_list, loss_fn)
            iou_list_cpu = [iou.cpu() for iou in iou_list]

        assert len(individual_iou_lists) == len(model_list), 'Individual iou lists length does not match model list length'
        assert len(individual_iou_lists[0]) == 4, 'Individual iou list length does not match number of densities'
        for i, indiv_iou_list in enumerate(individual_iou_lists):
                results_table[i+1][-4:] = indiv_iou_list        

if incremental_ensemble_size:
    print('Making plot of overall IoU for incremental ensemble')
    plotting_vals = []
    # plot the results of the incremental ensemble, with one line for ascending  and one for descending. x-axis is the number of models in the ensemble and y-axis is the IoU
    num_models = np.arange(1, len(model_list)+1)
    colors = {'ascending': 'darkgoldenrod', 'descending': 'darkgreen'}
    markers = {'ascending': 'o', 'descending': 'o'}
    fig, ax = plt.subplots()
    for key in incremental_results:
        iou_list = incremental_results[key]['iou_list']
        for i, iou in enumerate(iou_list):
            plotting_vals.append(iou[3].item())
            ax.plot(num_models[i], iou[3], marker=markers[key], color=colors[key], markersize=14)
    ax.set_xlabel('Number of models, N', fontsize=14)
    ax.set_ylabel('IoU', fontsize=14)
    ax.set_title('Test IoU scores for Varying Ensemble Sizes', fontsize=16 )
    ax.grid(alpha=0.5)
    handles = [
        # plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['ascending'],
        #             markersize=14),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['descending'],
                    markersize=14)
    ]
    ax.legend(handles=handles, labels=['Descending',
                                        # 'Ascending'
                                        ], loc='lower right', fontsize=12)
    ax.set_xticks(num_models)
    plt.tight_layout()
    plt.savefig(f"results_tables/{input}_ensemble_plot.png", dpi=300, bbox_inches='tight')
    plt.clf()
    print('Values plotted for input {}: {}'.format(input, plotting_vals))

    # print('Making plot of density specific IoU for incremental ensemble')   
    # fig, axs = plt.subplots(4,1, figsize=(10, 15))
    # for key in incremental_results:
    #     iou_list = incremental_results[key]['iou_list']
    #     for i, iou in enumerate(iou_list):
    #         axs[0].plot(num_models[i], iou[0], marker=markers[key], color=colors[key])
    #         axs[1].plot(num_models[i], iou[1], marker=markers[key], color=colors[key])
    #         axs[2].plot(num_models[i], iou[2], marker=markers[key], color=colors[key])
    #         axs[3].plot(num_models[i], iou[3], marker=markers[key], color=colors[key])
    # axs[3].set_title('Overall IoU')
    # axs[0].set_title('Heavy Density IoU')
    # axs[1].set_title('Medium Density IoU')
    # axs[2].set_title('Light Density IoU')

    # fig.supylabel('IoU', fontsize=14)
    # # axs[3].set_ylabel('Overall IoU')
    # # axs[0].set_ylabel('Heavy Density IoU')
    # # axs[1].set_ylabel('Medium Density IoU')
    # # axs[2].set_ylabel('Light Density IoU')

    # for ax in axs.flat:
    #     ax.grid(alpha=0.5)
    #     ax.set_xticks(num_models)
    
    # axs[3].legend(handles=handles, labels=['Descending', 'Ascending',], loc='lower right', fontsize=12)
    # fig.suptitle('Test IoU (by density) scores for Varying Ensemble Sizes', fontsize=16)
    # fig.supxlabel('Number of models, N', fontsize=14)
    # # fig.supylabel('IoU', fontsize=14)     
    # plt.tight_layout()
    # plt.savefig(f"results_tables/{input}_ensemble_plot_alldensities.png", dpi=300, bbox_inches='tight')

print(tabulate(results_table,headers="firstrow", tablefmt='grid'))
pickle.dump(results_table, open( "results_tables/{}.pkl".format(input), "wb" )) 

# remove the exp_num, high medium and low iou from the results table and print in latex format
latex_results_table = [row[0:3] + row[4:] for row in results_table]
print(tabulate(latex_results_table,headers="firstrow", tablefmt='latex_raw'))
