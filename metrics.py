import torch

def compute_iou(pred, true, level, iou_dict, print_ious=False, convert_to_classes=True):
    if convert_to_classes: # check if the preds were already converted to classes
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
    intersection = (pred + true == 2).sum()
    union = (pred + true >= 1).sum()
    iou = intersection / union
    # iou_dict[level]['prev_int'] = intersection
    # iou_dict[level]['prev_union'] = union
    if torch.isnan(iou) == False:
        iou_dict[level]['int'] += intersection
        iou_dict[level]['union'] += union
        if print_ious: print('{} density smoke gives: {} IoU'.format(level, iou))
        return iou_dict
    else:
        return iou_dict

def get_iou_by_density(iou_dict):
    try:
        high_iou = iou_dict['high']['int']/iou_dict['high']['union']
    except ZeroDivisionError:
        high_iou = float('nan') # np.nan
        
    try:
        med_iou = iou_dict['medium']['int']/iou_dict['medium']['union']
    except ZeroDivisionError:
        med_iou = float('nan')
        
    try:
        low_iou = iou_dict['low']['int']/iou_dict['low']['union']
    except ZeroDivisionError:
        low_iou = float('nan')
        
    try:
        iou = (iou_dict['high']['int'] + iou_dict['medium']['int'] + iou_dict['low']['int'])/(iou_dict['high']['union'] + iou_dict['medium']['union'] + iou_dict['low']['union']) # current overall IoU calculation
        
    except ZeroDivisionError:
        iou = float('nan')

    return [high_iou, med_iou, low_iou, iou]

def get_weighted_iou(iou_dict, dn_weights):
    [high_iou, med_iou, low_iou, _] = get_iou_by_density(iou_dict)
    weighted_iou = dn_weights[0]*high_iou + dn_weights[1]*med_iou + dn_weights[2]*low_iou
    return weighted_iou

def display_iou(iou_dict):
    [high_iou, med_iou, low_iou, iou] = get_iou_by_density(iou_dict)
    
    print('OVERALL HIGH DENSITY SMOKE GIVES: {} IoU'.format(high_iou))
    print('OVERALL MEDIUM DENSITY SMOKE GIVES: {} IoU'.format(med_iou))
    print('OVERALL LOW DENSITY SMOKE GIVES: {} IoU'.format(low_iou))
    print('OVERALL OVER ALL DENSITY GIVES: {} IoU'.format(iou))

    return [high_iou, med_iou, low_iou, iou]

