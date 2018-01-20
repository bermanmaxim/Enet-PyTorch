# ENet-PyTorch
Fork based on [convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch) and adapted for image segmentation with [ENet](https://github.com/e-lab/ENet-training) in PyTorch.

## Conversion from torch ENet
* Convert the source Enet trained model in PyTorch from GPU to CPU weights

```bash
$ th convert_torch_gpu_to_cpu.lua -model model-best.net
saved cpu-model-best.net
```

* Convert the lua model to a pytorch script and state dict.

```bash
$ python convert_torch.py --model cpu-model-best.net --output enet_pytorch
```
This produces a standalone script `enet_pytorch.py` defining a `torch.nn` module, and an associated state dict.

*Example usage:*
```python
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from enet_pytorch import enet_pytorch

enet_pytorch.load_state_dict(torch.load('enet_pytorch.pth'))
enet_pytorch.cuda().eval()

img = 'frankfurt_000001_030067_leftImg8bit.png'
im = cv2.imread(img).astype(np.float32)[:, :, ::-1]/255
inp = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).cuda()
inp = F.upsample(Variable(inp), (512, 1024), mode='bilinear').data

out = enet_pytorch(Variable(inp))
```

Tested to reproduce ENet original results. For reference the output on the author's cityscapes model is included as `enet_pytorch.py`.

## Implementation notes
The max-unpooling modules unpool the last pooling module (FIFO), which is appropriate for encoder-decoder segmentation networks. This converter is not tested for other networks than ENet.


## References
* Refer to [ENet](https://github.com/e-lab/ENet-training) original authors.
* Based on [carwin](https://github.com/clcarwin)'s [convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch).

## License
In compliance with ENet's license, this work follows the [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/) license.

