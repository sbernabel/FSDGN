

from basicsr.models import build_model
from basicsr.utils import tensor2img
import cv2
import torch

opt = {}
opt['model_type'] = 'DehazeModel'
opt['num_gpu'] = 0
opt['scale'] = 1
opt['manual_seed'] = 10
opt['is_train'] = False
opt['network_g'] = {'type': 'FSDGN'}
opt['dist'] = False
opt['path'] = {
    'pretrain_network_g': './pretrained models/net_g_latest.pth',
}


model = build_model(opt)

filename = '0012.jpg'
img = cv2.imread(f'./input/{filename}')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

data = {'lq': img}
model.feed_data(data)
model.test()
visuals = model.get_current_visuals()
pred = tensor2img([visuals['result']])
cv2.imwrite(f'./output/fsdgn_{filename}', pred)
