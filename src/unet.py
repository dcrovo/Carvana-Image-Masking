"""
Developed by Daniel Crovo

"""

import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv == nn.Sequential(
            # Conv1 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Conv2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.conv(X)
    
class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels =1, feature_maps=[64, 128, 256, 512]): #assuming an
        super(Unet, self).__init__()
        self.downs = nn.ModuleList() # Creates a list of operations 
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # red arrows

        # CNN (downs): 
        for feat in feature_maps:
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat # update input channels 
        # Bottleneck:
        self.bottleneck = DoubleConv(feature_maps[-1], feature_maps[-1]*2)

        #DCNN (ups):

        for feat in reversed(feature_maps):
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2)) # green arrows (Unet architecture)
            self.ups.append(DoubleConv(feat*2, feat))

        # Output:
        self.final_conv = nn.Conv2d(feature_maps[0], out_channels, kernel_size=1) # Cyan arrows