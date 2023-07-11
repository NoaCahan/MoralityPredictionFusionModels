import torch
from torch import nn
import torch.nn.functional as F
from models.sanet.attention_blocks import ( AttentionModule1, 
                                      AttentionModule2, 
                                      AttentionModule3,  
                                      AttentionModule4 )
from models.sanet.densenet import densenet121
import math

class SANet(nn.Module):

    def __init__(self,
                 sample_duration,
                 sample_size,
                 num_classes = 2):

        super(SANet, self).__init__()

        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        self.base_model = densenet121(
                            num_classes = self.num_classes,
                            sample_size = self.sample_size,
                            sample_duration = self.sample_duration)

        self.attention1 = AttentionModule1(256, 256)
        self.attention2 = AttentionModule2(512, 512)
        self.attention3 = AttentionModule3(1024, 1024)
        self.attention4 = AttentionModule4(1024, 1024)

        self.dropout = nn.Dropout3d(p=0.8, inplace=False)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):

        x = self.base_model.features.conv0(x)
        x = self.base_model.features.norm0(x)
        x = self.base_model.features.relu0(x)
        x = self.base_model.features.pool0(x)
        x = self.base_model.features.denseblock1(x)
        x = self.attention1(x)
        x = self.base_model.features.transition1(x)     
        x = self.base_model.features.denseblock2(x)
        x = self.attention2(x)
        x = self.base_model.features.transition2(x)
        x = self.base_model.features.denseblock3(x)
        x = self.attention3(x)
        x = self.base_model.features.transition3(x)
        x = self.base_model.features.denseblock4(x)
        # Uncomment this for 4 attention blocks
        #x = self.attention4(x)
        x = self.base_model.features.norm5(x)

        out = F.relu(x, inplace=True)
        last_duration = int(math.ceil(self.base_model.sample_duration / 16))
        last_size = int(math.floor(self.base_model.sample_size / 32))
        out = F.avg_pool3d( out, kernel_size=(last_duration, last_size, last_size))
        out = self.dropout(out)        
        out = out.view( x.size(0), -1)

        # Comment for multimodality
        emb = out
        out = self.base_model.classifier(out)

        ## Experimennt with embedding size for multimodality
        #emb = self.fc1(out)
        #out = self.fc2(emb)

        return out, emb