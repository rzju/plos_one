import numpy as np

def dice_coefficient_multiclass(y_true, y_pred, num_classes, smooth=1e-6):
    dice_array = np.zeros(num_classes)
    for i in range(num_classes):
        y_true_f = (y_true == i).float().contiguous().view(-1)
        y_pred_f = (y_pred == i).float().contiguous().view(-1)
        intersection = (y_true_f * y_pred_f).sum()
        dice_array[i] = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
    return dice_array

def jaccard_index(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.contiguous().view(-1)
    y_pred_f = y_pred.contiguous().view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def calculate_metrics(outputs, masks):
    preds = torch.argmax(outputs, dim=1)
    if masks.dim() > 1:
        masks = masks.argmax(dim=1)
    dice = dice_coefficient(preds, masks)
    jaccard = jaccard_index(preds, masks)
    return dice.item(), jaccard.item()