import torch
import torch.nn as nn
from collections import OrderedDict

ACTIVATION = nn.ReLU6 

class Add(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return a + b

class Concatenate(nn.Module):   
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, *x):
        return torch.cat(x, dim=self.dim)

def autopad(k, p=None, d=1):  
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p
    
class ConvBNReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, padding=None, activation=ACTIVATION):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=autopad(kernel_size, padding, 1),
                              bias=False)
        self.bn = nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.03)
        self.activation = activation(inplace=True) if activation is not None else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, expansion=1.0, shortcut=True):
        super().__init__()
        hidden_channels = int(output_channels * expansion)
        self.conv1 = ConvBNReLU(input_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBNReLU(hidden_channels, output_channels, kernel_size=3, padding=1)
        self.shortcut = shortcut and input_channels == output_channels
        if self.shortcut:
            self.add = Add()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        if self.shortcut:
            x = self.add(x, inputs)
        return x

class CSP(nn.Module):
    def __init__(self, input_channels, output_channels, n=1, expansion=0.5, shortcut=True):
        super().__init__()
        hidden_channels = int(output_channels * expansion)
        self.conv1 = ConvBNReLU(input_channels, hidden_channels, kernel_size=1, padding=0)
        self.blocks = nn.Sequential(*(Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut) for _ in range(n)))
        self.conv2 = ConvBNReLU(input_channels, hidden_channels, kernel_size=1, padding=0)
        self.concat = Concatenate(dim=1)
        self.conv3 = ConvBNReLU(2 * hidden_channels, output_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1 = self.blocks(self.conv1(inputs))
        x2 = self.conv2(inputs)
        return self.conv3(self.concat(x1, x2))

class SPPF(nn.Module):
    def __init__(self, in_channel, out_channel, k=5):
        super().__init__()
        hidden_channel = in_channel // 2
        self.conv1 = ConvBNReLU(input_channels=in_channel, output_channels=hidden_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.concat = Concatenate(dim=1)
        self.conv2 = ConvBNReLU(input_channels=hidden_channel * 4, output_channels=out_channel)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(self.concat(x, y1, y2, y3))
    
class Backbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv = ConvBNReLU(3, configs['input_channel'], kernel_size=6, stride=2, padding=2)
        self.block_0 = self.make_block(*configs['block_0'])
        self.block_1 = self.make_block(*configs['block_1'])
        self.block_2 = self.make_block(*configs['block_2'])
        self.block_3 = self.make_block(*configs['block_3'])
        self.sppf = SPPF(*configs['sppf'])
        
    def make_block(self, input_channels, output_channels, n):
        return nn.Sequential(OrderedDict([
            ('conv', ConvBNReLU(input_channels, output_channels, kernel_size=3, stride=2, padding=1)),
            ('csp', CSP(output_channels, output_channels, n=n))
        ]))

    def forward(self, inputs):
        x = self.block_1(self.block_0(self.conv(inputs))) # P3 (8x) 不再作为输出
        p4 = self.block_2(x)      # 16x
        p5 = self.sppf(self.block_3(p4)) # 32x
        return p4, p5

class Proto(nn.Module):
    def __init__(self, c1, c_=256, c2=32): 
        super().__init__()
        self.cv1 = ConvBNReLU(c1, c_, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
	self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = ConvBNReLU(c_, c_, kernel_size=3, padding=1)
        self.cv3 = ConvBNReLU(c_, c2)

    def forward(self, x):
        x = self.cv1(x)
        x = self.upsample1(x)
	x = self.upsample2(x)
        x = self.cv2(x)
        x = self.cv3(x)
        return x
    
class SegmentHead(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.nc, self.nm, self.npr = configs['num_class'], configs['num_mask'], configs['num_proto']
        self.no = 5 + self.nc + self.nm     
        self.nl = 2 # 仅 P4, P5
        self.na = len(configs['anchors'][0])
        self.stride = [16., 32.]
        
        self.register_buffer('anchors', torch.tensor(configs['anchors']).float().view(self.nl, -1, 2))
        self.head = nn.ModuleList(nn.Conv2d(ch, self.no * self.na, 1) for ch in configs['channel'])
        self.proto = Proto(configs['channel'][0], self.npr, self.nm) # 输入为 P4_out
        
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]

    def forward(self, p4, p5):
        p = self.proto(p4)
        medium_out = self.head[1](p4)
        large_out = self.head[2](p5)
        if self.eval_mode:
            return medium_out, large_out, p 
        

        z = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            
            xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.nm), dim=4)
            xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]
            wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]
            y = torch.cat((xy, wh, conf.sigmoid(), mask), dim=4)
            z.append(y.view(bs, -1, self.no))

        return torch.cat(z, 1), p, x

    def _make_grid(self, nx, ny, i):
        y, x = torch.arange(ny, device=self.anchors.device), torch.arange(nx, device=self.anchors.device)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(1, self.na, ny, nx, 2) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.na, 1, 1, 2).expand(1, self.na, ny, nx, 2)
        return grid, anchor_grid

class FPN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 仅保留 P4, P5 融合层
        self.conv_p5 = ConvBNReLU(configs[0][0], configs[0][1]) # 下采样 P5 降维
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concatenate(dim=1)
        self.csp_p4 = CSP(configs[0][1] + configs[0][0]//2, configs[0][2], n=configs[0][3], shortcut=False)
        
        # 自底向上
        self.down_p4 = ConvBNReLU(configs[0][2], configs[0][2], kernel_size=3, stride=2, padding=1)
        self.csp_p5 = CSP(configs[0][2] + configs[0][1], configs[0][1]*2, n=configs[0][3], shortcut=False)

    def forward(self, p4, p5):
        p5_feat = self.conv_p5(p5)
        x = self.upsample(p5_feat)
        p4_out = self.csp_p4(self.concat(x, p4))
        
        x = self.down_p4(p4_out)
        p5_out = self.csp_p5(self.concat(x, p5_feat))
        return p4_out, p5_out

class Yolov5(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        if configs is None: configs = get_configs()
        self.backbone = Backbone(configs['backbone'])
        self.fpn = FPN(configs['fpn'])
        self.head = SegmentHead(configs['head'])

    def forward(self, x):
        p4, p5 = self.backbone(x)
        p4_f, p5_f = self.fpn(p4, p5)
        return self.head(p4_f, p5_f)

def get_configs(depth=0.33, width=0.25, nc=80):
    w = lambda x: int(x * width)
    d = lambda x: round(x * depth)
    return {
        'backbone': {
            'input_channel': 17,
            'block_0': [17, w(128), d(3)],
            'block_1': [w(128), w(256), d(6)],
            'block_2': [w(256), w(512), d(9)],
            'block_3': [w(512), w(1024), d(3)],
            'sppf': [w(1024), w(1024)]
        },
        'fpn': [
            [w(1024), w(512), w(512), d(3), False], # P5 to P4 config
        ],
        'head': {
            'channel': [w(512), w(1024)], # P4_out, P5_out
            'num_class': nc,
            'anchors': [
                [[1.72, 3.21], [3.98, 2.97], [3.44, 7.20]], # 原 P4 anchors
                [[3.70, 3.13], [4.18, 6.80], [10.1, 8.32]]  # 原 P5 anchors
            ],
            'num_mask': 32,
            'num_proto': 64
        }
    }

if __name__ == "__main__":
    model = Yolov5()
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    output, proto, raw = model(dummy_input)
    print(f"Output shape: {output.shape}") # 应为 [1, 2400, 5+nc+32] (1200=20x20x3 + 40x40x3)
    print(f"Proto shape: {proto.shape}")   # 应为 [1, 32, 160, 160] (4x upsample from 40x40)