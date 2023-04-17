import glob
import os
from mmengine import ProgressBar
import mmengine

folder = 'cache_data/clip_data'
files = glob.glob(os.path.join(folder, '*.pkl'))
par = ProgressBar(len(files))
for file in files:
    par.update()
    data = mmengine.load(file)
    for k, v in data.items():
        if isinstance(v, mmengine.Tensor):
            data[k] = v.cpu()
    mmengine.dump(data, file)

