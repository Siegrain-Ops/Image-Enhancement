import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

#channel attention module for cbam
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False) #compress channels
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False) #decompress channels
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)  
        max_out = torch.amax(x, dim=(2, 3), keepdim=True)  

        #make sure they have same dim
        avg_out = self.fc1(avg_out)
        max_out = self.fc1(max_out)

        out = avg_out + max_out  #now they have the same dim
        out = self.fc2(F.relu(out))
        return x * torch.sigmoid(out)



#spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out = torch.amax(x, dim=1, keepdim=True)  

        #it concatenates along the dim of the channels (1) and then do a conv
        out = torch.cat([avg_out, max_out], dim=1)  #now they have same dim
        out = self.conv(out)
        return x * torch.sigmoid(out)



#CBAM module (channel + spatial attention)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


#residual block with skip connection
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual #skip connection
        return out


#enhancement CNN model
class EnhancementCNN(nn.Module):
    def __init__(self, in_channels=3, num_residual_blocks=5):
        super(EnhancementCNN, self).__init__()
        
        #feature extraction block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        #CBAM attention
        self.cbam = CBAM(64)

        #enhancement block (multiple residual block + skip connections)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])

        #refinement block 
        self.conv2 = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.cbam(x1) #apply attention
        x2 = self.res_blocks(x1) #enhancement
        x2 += x1 #skip connections
        self.activation = nn.Sigmoid() 
        out = self.activation(self.conv2(x2))

        return out


#test model
if __name__ == "__main__":
    model = EnhancementCNN()
    test_input = torch.randn(1, 3, 512, 512)  #Example 256x256 RGB image
    output = model(test_input)
    print("Output Shape:", output.shape)