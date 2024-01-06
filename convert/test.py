import torch
kpts = torch.randn(1, 1024, 2)
print(kpts)

width = 640
height = 480

one = kpts.new_tensor(1)
size = torch.stack([one*width, one*height])[None]
center = size / 2
print("Center: ", center)
scaling = size.max(1, keepdim=True).values * 0.7
print("scaling: ", scaling)
result = (kpts - center[:, None, :]) / scaling[:, None, :]

print('result: ', result)
print(type(result[0][0][0].item()))