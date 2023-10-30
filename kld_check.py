import torch
import torch.nn.functional as F

kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
x=torch.randn(2, 2, requires_grad=True)
print(x)
input = F.log_softmax(x, dim=1)
print(input)
print()

y=torch.randn(2, 2)
print(y)
target = F.softmax(y, dim=1)
print(target)
print()

output = kl_loss(input, target)
print(output)



kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
log_target = F.log_softmax(y, dim=1)
output = kl_loss(input, log_target)