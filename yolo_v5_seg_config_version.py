import os
import torch
import torch.nn as nn
from collections import OrderedDict
import pdb

ACTIVATION = nn.ReLU6  # nn.SiLU


class Add(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, a, b):
        return a+b


class Concatenate(nn.Module):   
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim = self.dim)

def autopad(k, p=None, d=1):  
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p
    
class ConvBNReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 1, stride = 1, padding = None, activation = ACTIVATION):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = input_channels,
                              out_channels = output_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = autopad(kernel_size, padding, 1),
                              bias = False)
        self.bn = nn.BatchNorm2d(output_channels, eps = 1e-3, momentum = 0.03)
        if activation is not None:
            self.activation = activation(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        inputs = x
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
        

class Bottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, expansion = 1.0, shortcut = True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.expansion = expansion
        hidden_channels = int(output_channels * expansion)
        self.conv1 = ConvBNReLU(input_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBNReLU(hidden_channels, output_channels, kernel_size=3, padding=1)
        self.shortcut = shortcut
        if self.shortcut:
            self.add = Add()

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x = self.add(x, inputs)
        return x


class CSP(nn.Module):
    def __init__(self, input_channels, output_channels, n = 1, expansion = 0.5, shortcut = True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.expansion = expansion
        hidden_channels = int(output_channels * expansion)
        
        self.conv1 = ConvBNReLU(input_channels, hidden_channels, kernel_size=1, padding=0)

        self.block_repeats = n
        self.shortcut = shortcut
        self.blocks = nn.Sequential(*(Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut) for _ in range(n)))
        
        self.conv2 = ConvBNReLU(input_channels, hidden_channels, kernel_size=1, padding=0)
        self.concat = Concatenate(dim=1)
        self.conv3 = ConvBNReLU(2*hidden_channels, output_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.blocks(x1)
        x2 = self.conv2(inputs)
        x = self.concat(x1, x2)
        x = self.conv3(x)
        return x


class SPPF(nn.Module):
    def __init__(self, in_channel, out_channel, k = 5):
        super().__init__()
        hidden_channel = in_channel//2
        self.conv1 = ConvBNReLU(input_channels = in_channel, output_channels = hidden_channel)
        self.maxpool1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.concat = Concatenate(dim=1)
        self.conv2 = ConvBNReLU(input_channels = hidden_channel*4, output_channels = out_channel)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool1(x)
        y2 = self.maxpool2(y1)
        y3 = self.maxpool3(y2)
        x = self.concat(x,y1,y2,y3)
        x = self.conv2(x)
        return x
    
    
class Backbone(nn.Module):
    def __init__(self, configs, residual_block = Bottleneck):
        super().__init__()
        # Setting
        self.configs = configs
        self.residual_block = residual_block

        # 1. Input block
        self.conv = ConvBNReLU(input_channels = 3, 
                               output_channels = configs['input_channel'],
                               kernel_size = 6,
                               stride = 2,
                               padding = 2)
        
        # 2. Main blocks conv+csp
        self.block_0 = self.make_block(*configs['block_0'])
        self.block_1 = self.make_block(*configs['block_1'])
        self.block_2 = self.make_block(*configs['block_2'])
        self.block_3 = self.make_block(*configs['block_3'])

        # 3. SPPF block
        self.sppf = SPPF(*configs['sppf'])
        
    def make_block(self, input_channels, output_channels, n):
        modules = OrderedDict()
        modules['conv'] = ConvBNReLU(input_channels = input_channels,
                                     output_channels = output_channels,
                                     kernel_size = 3, stride = 2, padding = 1)
        modules['csp'] = CSP(input_channels = output_channels,
                             output_channels = output_channels,
                             n = n,
                             )

        return nn.Sequential(modules)

    def forward(self, inputs):
        x = self.conv(inputs)    
        x = self.block_0(x)      
        x = self.block_1(x)     
        p3 = x
        x = self.block_2(x)      
        p4 = x
        x = self.block_3(x)      
        x = self.sppf(x)         
        p5 = x

        return p3, p4, p5

class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = ConvBNReLU(input_channels = c1, output_channels = c_, kernel_size = 3, padding = 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = ConvBNReLU(input_channels = c_, output_channels = c_, kernel_size = 3) # padding=1
        self.cv3 = ConvBNReLU(input_channels = c_, output_channels = c2)

    def forward(self, x):
        x = self.cv1(x)
        x = self.upsample(x)
        x = self.cv2(x)
        x = self.cv3(x)
        return x
    
class SegmentHead(nn.Module):
    stride = None
    # YOLOv5 Segment head for segmentation models
    def __init__(self, configs):
        super().__init__()
        nc = configs['num_class']
        anchors = configs['anchors']
        channel_configs = configs['channel']
        nm = configs['num_mask']
        npr = configs['num_proto']
        self.nm = nm          
        self.npr = npr        
        self.nc = nc          
        self.no = 5+nc+nm     
        
        self.nl = len(anchors)         
        self.na = len(anchors[0]) 
        
        self.grid = [torch.empty(0) for _ in range(self.nl)]  
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)] 
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))

        assert len(channel_configs) == self.nl, 'number of level should match number of input channel in head config'
        self.in_channel_list = channel_configs
        self.out_channel = self.no*self.na  
        self.head = nn.ModuleList(nn.Conv2d(ch, self.out_channel, 1) for ch in self.in_channel_list)

        self.proto = Proto(self.in_channel_list[0], self.npr, self.nm)  
        
        self.eval_mode = False

    def forward(self, p3, p4, p5):
        p = self.proto(p3)
        small_out = self.head[0](p3)
        medium_out = self.head[1](p4)
        large_out = self.head[2](p5)
        if self.eval_mode:
            return small_out, medium_out, large_out, p 

        x = [small_out, medium_out, large_out]
        z = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape 
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:           
                y = self._get_inference_output(i, x[i])
                z.append(y)

        if self.training:
            return x, p
        else:
            return torch.cat(z, 1), p, x

    def _get_inference_output(self, i, x):
        bs, _, ny, nx, _ = x.shape
        if self.grid[i].shape[2:4] != x.shape[2:4]:
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
        xy, wh, conf, mask = x.split((2, 2, self.nc + 1, self.nm), dim = 4)
        xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  
        wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]   
        y = torch.cat((xy, wh, conf.sigmoid(), mask), dim = 4)     
        y = y.view(bs, self.na * nx * ny, self.no)
        return y

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') 
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5 
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    

class FPN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv1 = ConvBNReLU(input_channels = configs[0][0], output_channels = configs[0][1])
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat1 = Concatenate(dim=1)
        self.csp = CSP(input_channels = configs[0][1]*2, output_channels = configs[0][2],
                       n = configs[0][3], shortcut = configs[0][4])
        
        self.conv2 = ConvBNReLU(input_channels = configs[1][0], output_channels = configs[1][1])
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat2 = Concatenate(dim=1)
        self.csp_p3 = CSP(input_channels = configs[1][1]*2, output_channels = configs[1][2],
                          n = configs[1][3], shortcut = configs[0][4])
        
        self.conv3 = ConvBNReLU(input_channels = configs[2][0],
                                output_channels = configs[2][1],
                                kernel_size = 3, stride = 2, padding = 1)
        self.concat3 = Concatenate(dim=1)
        self.csp_p4 = CSP(input_channels = configs[2][1]*2, output_channels = configs[2][2],
                          n = configs[2][3], shortcut = configs[0][4])
        
        self.conv4 = ConvBNReLU(input_channels = configs[3][0],
                                output_channels = configs[3][1],
                                kernel_size = 3, stride = 2, padding = 1)
        self.concat4 = Concatenate(dim=1)
        self.csp_p5 = CSP(input_channels = configs[3][1]*2, output_channels = configs[3][2],
                          n = configs[3][3], shortcut = configs[0][4])
        

    def forward(self, p3, p4, p5):
        p5 = self.conv1(p5)      
        x = self.upsample1(p5)   
        x = self.concat1(x, p4)  
        x = self.csp(x)        

        S3 = self.conv2(x)       
        x = self.upsample2(S3)   
        x = self.concat2(x, p3)  
        x = self.csp_p3(x)      
        p3_out = x 
       
        x = self.conv3(x)        
        x = self.concat3(x, S3)  
        x = self.csp_p4(x)       
        p4_out = x

        x = self.conv4(x)        
        x = self.concat4(x, p5) 
        x = self.csp_p5(x)      
        p5_out = x

        return p3_out, p4_out, p5_out
        
    
class Yolov5(nn.Module):
    stride = torch.tensor([ 8., 16., 32.])
    def __init__(self, configs=None):
        if configs is None:
            configs = get_configs()
        super().__init__()
        self.backbone = Backbone(configs['backbone'])
        self.fpn = FPN(configs['fpn'])
        self.head = SegmentHead(configs['head'])
        self.head.stride = self.stride

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3_out,p4_out,p5_out = self.fpn(p3,p4,p5)
        return self.head(p3_out,p4_out,p5_out)


def get_configs(depth=0.33, width=0.25, nc = 80):
    configs = {'backbone': {'input_channel': 17, 
                            'block_0': [17, int(128*width), round(3*depth)], 
                            'block_1': [int(128*width), int(256*width), round(6*depth)],
                            'block_2': [int(256*width), int(512*width), round(9*depth)],
                            'block_3': [int(512*width), int(1024*width), round(3*depth)],
                            'sppf': [int(1024*width), int(1024*width)],
                            },
               'fpn': [[int(1024*width),int(512*width),int(512*width),round(3*depth), False], # conv1, csp
                       [int(512*width),int(256*width),int(128*width),round(3*depth), False],  # conv2, csp_p3
                       [int(128*width),int(256*width),int(512*width),round(3*depth), False],  # conv3, csp_p4
                       [int(512*width),int(512*width),int(1024*width),round(3*depth), False], # conv4, csp_p5
                       ],
               'head': {'channel': [int(128*width),int(512*width),int(1024*width)],
                        'num_class': nc,
                        'anchors': [[[ 1.08917,  1.18461],[ 1.54856,  3.34862],[ 3.82294,  2.45488]],
                                    [[ 1.72693,  3.21882],[ 3.98780,  2.97551],[ 3.44934,  7.20848]],
                                    [[ 3.70934,  3.13941],[ 4.18038,  6.80048],[10.10490,  8.32616]]],
                        'num_mask': 32,
                        'num_proto': 64,
                        }, 
               }
    return configs


configure_list = [{'quant_types': ['weight', 'input', 'output'],
                   'quant_bits': {'weight': 8, 'input': 8, 'output': 8},
                   'op_names': ['backbone.conv.conv',]},
                   
                  {'quant_types': ['weight', 'output'],
                   'quant_bits': {'weight': 8, 'output': 8},
                   'op_names': ['backbone.block_0.conv.conv',
                                'backbone.block_0.csp.conv1.conv', 'backbone.block_0.csp.conv2.conv', 'backbone.block_0.csp.conv3.conv',
                                'backbone.block_0.csp.blocks.0.conv1.conv', 'backbone.block_0.csp.blocks.0.conv2.conv',
                                'backbone.block_1.conv.conv',
                                'backbone.block_1.csp.conv1.conv', 'backbone.block_1.csp.conv2.conv', 'backbone.block_1.csp.conv3.conv',
                                'backbone.block_1.csp.blocks.0.conv1.conv', 'backbone.block_1.csp.blocks.0.conv2.conv',
                                'backbone.block_1.csp.blocks.1.conv1.conv', 'backbone.block_1.csp.blocks.1.conv2.conv',
                                'backbone.block_2.conv.conv',
                                'backbone.block_2.csp.conv1.conv', 'backbone.block_2.csp.conv2.conv', 'backbone.block_2.csp.conv3.conv',
                                'backbone.block_2.csp.blocks.0.conv1.conv', 'backbone.block_2.csp.blocks.0.conv2.conv',
                                'backbone.block_2.csp.blocks.1.conv1.conv', 'backbone.block_2.csp.blocks.1.conv2.conv',
                                'backbone.block_2.csp.blocks.2.conv1.conv', 'backbone.block_2.csp.blocks.2.conv2.conv',
                                'backbone.block_3.conv.conv',
                                'backbone.block_3.csp.conv1.conv', 'backbone.block_3.csp.conv2.conv', 'backbone.block_3.csp.conv3.conv',
                                'backbone.block_3.csp.blocks.0.conv1.conv', 'backbone.block_3.csp.blocks.0.conv2.conv',
                                'backbone.sppf.conv1.conv', 'backbone.sppf.conv2.conv',
                                'fpn.conv1.conv',
                                'fpn.csp.conv1.conv', 'fpn.csp.conv2.conv', 'fpn.csp.conv3.conv',
                                'fpn.csp.blocks.0.conv1.conv', 'fpn.csp.blocks.0.conv2.conv',
                                'fpn.conv2.conv',
                                'fpn.csp_p3.conv1.conv', 'fpn.csp_p3.conv2.conv', 'fpn.csp_p3.conv3.conv',
                                'fpn.csp_p3.blocks.0.conv1.conv', 'fpn.csp_p3.blocks.0.conv2.conv',
                                'fpn.conv3.conv',
                                'fpn.csp_p4.conv1.conv', 'fpn.csp_p4.conv2.conv', 'fpn.csp_p4.conv3.conv',
                                'fpn.csp_p4.blocks.0.conv1.conv', 'fpn.csp_p4.blocks.0.conv2.conv',
                                'fpn.conv4.conv',
                                'fpn.csp_p5.conv1.conv', 'fpn.csp_p5.conv2.conv', 'fpn.csp_p5.conv3.conv',
                                'fpn.csp_p5.blocks.0.conv1.conv', 'fpn.csp_p5.blocks.0.conv2.conv',
                                ]},
                  
                  {'quant_types': ['output'],
                   'quant_bits': {'output': 8},
                   'op_names': ['backbone.block_0.csp.concat',
                                'backbone.block_0.csp.blocks.0.add',
                                'backbone.block_1.csp.concat',
                                'backbone.block_1.csp.blocks.0.add',
                                'backbone.block_1.csp.blocks.1.add',
                                'backbone.block_2.csp.concat',
                                'backbone.block_2.csp.blocks.0.add',
                                'backbone.block_2.csp.blocks.1.add',
                                'backbone.block_2.csp.blocks.2.add',
                                'backbone.block_3.csp.concat',
                                'backbone.block_3.csp.blocks.0.add',
                                'backbone.sppf.maxpool1',
                                'backbone.sppf.maxpool2',
                                'backbone.sppf.maxpool3',
                                'backbone.sppf.concat',
                                'fpn.concat1', 'fpn.csp.concat',
                                'fpn.concat2', 'fpn.csp_p3.concat', 
                                'fpn.concat3', 'fpn.csp_p4.concat',
                                'fpn.concat4', 'fpn.csp_p5.concat',
                                ]},
                  
                  {'quant_types': ['weight', 'output'],
                   'quant_bits': {'weight': 8, 'output': 8},
                   'op_names': ['head.head.0', 'head.head.1', 'head.head.2']},

                  {'quant_types': ['weight', 'output'],
                   'quant_bits': {'weight': 8, 'output': 8},
                   'op_names': ['head.proto.cv1.conv', 'head.proto.cv2.conv', 'head.proto.cv3.conv']},

                  ]

    
if __name__ == "__main__":
    # from torchsummaryX import summary
    
    model = Yolov5(configs=get_configs(depth=0.33, width=0.25, nc=80)) # 0.33 0.32
    model.eval()
    model(torch.ones((1, 3, 640, 640)))
    # summary(model, torch.ones((1, 3, 640, 640)))
    pdb.set_trace()
