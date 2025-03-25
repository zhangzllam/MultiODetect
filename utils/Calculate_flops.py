from torchvision.models import resnet18
from CoreNetwork import amount_net
import torch
from thop import profile

# #call Transception_res
model = resnet18()
input = torch.randn(128, 3, 112, 112) ### ？
Flops, params = profile(model, inputs=(input,)) # macs
print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值



# #call Transception_res
model = amount_net(10)
input = torch.randn(128, 3, 112, 112) ### ？
Flops, params = profile(model, inputs=(input,)) # macs
print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值