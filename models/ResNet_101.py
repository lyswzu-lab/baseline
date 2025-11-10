import torch
import torchvision
import torch.nn as nn
import models.loss_fns as loss_fns

class model(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(model, self).__init__()
        # 1. 初始化模型，但设置 pretrained=False
        M = torchvision.models.resnet101(pretrained=False) 

        # --- [!! 开始修改 !!] ---
        # 2. 定义作者提供的权重文件路径
        file = r'/data/dsj/lys/SpliceMix-main/SpliceMix-main/ImageNet_ResNet101_SpliceMix_te79.912_E163.pth.tar'

        # 3. 加载自定义权重
        try:
            ckpt = torch.load(file, map_location='cpu')
            # 确保使用 'state_dict' 键来加载
            M.load_state_dict(ckpt['state_dict']) 
            del ckpt
            print(f"成功: 从 {file} 加载了作者的自定义ImageNet权重。")
        except FileNotFoundError:
            print(f"警告: 在 {file} 未找到自定义权重。模型将使用随机初始化。")
        except Exception as e:
            print(f"警告: 加载自定义权重 {file} 时出错: {e}。模型将使用随机初始化。")
        # --- [!! 结束修改 !!] ---
        self.backbone = nn.Sequential(M.conv1, M.bn1, M.relu, M.maxpool,
                                      M.layer1, M.layer2, M.layer3, M.layer4, )
        self.num_classes = num_classes

        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)

    def forward(self, inputs, args=None):  #

        fea4 = self.backbone(inputs)  # bs, C, h, w
        fea_gmp = self.glb_pooling(fea4).flatten(1)  # bs, C
        output = self.cls(fea_gmp)    # bs, nc

        return output

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.backbone.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.backbone.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

class Loss_fn(loss_fns.BCELoss):
    def __init__(self):
        super(Loss_fn, self).__init__()

if __name__ == '__main__':
    inputs = torch.randn((2, 3, 448, 448)).cuda()
    target = torch.zeros((2, 20)).cuda()
    target[:, 1:3] = 1

    loss_fn = Loss_fn()

    model = model(20).cuda()
    output = model(inputs)

    loss = loss_fn(output, target)
    loss.backward()

    a= 'pause'
