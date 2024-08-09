import torch
import torch.nn as nn
from torch import einsum

class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch_size, class_num, height, width)
        shp_y = gt.shape  # (batch_size, height, width)

        
        

        # one hot code for gt
        with torch.no_grad():
            if len(gt.shape) == 3:  # 如果gt是(batch_size, height, width)，则扩展为(batch_size, 1, height, width)
                gt = gt.unsqueeze(1)
            gt = gt.long()
            y_onehot = torch.zeros((gt.size(0), shp_x[1], gt.size(2), gt.size(3)), device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
    
        
      

        # 计算Generalized Dice Loss
      
        w: torch.Tensor = 1 / (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * einsum("bcxy,bcxy->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", y_onehot))
        divided: torch.Tensor = -2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()
       

        return gdc
