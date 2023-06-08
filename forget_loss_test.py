"""
Author:  Cax
File:    forget_loss_test
Project: T2I-Adapter
Time:    2023/6/8
Des:     
"""

import os
from ldm.metrics.clip import ClipImageFeature
from ldm.metrics.mmd import mmd

tar_path = 'C:\\Users\\cax11\\Desktop\\test\\continual_dog'
t1_path = [os.path.join(tar_path, str(i).zfill(2) + '.jpg') for i in range(5)]
t2_path = [os.path.join(tar_path, str(i).zfill(2) + '.jpg') for i in range(5, 11)]
t3_path = [os.path.join(tar_path, str(i).zfill(2) + '.jpg') for i in range(11, 15)]

clip = ClipImageFeature(pretrained_model_name_or_path='F:\\Clip-Vit-Large-Patch14')
t1 = clip(path=t1_path)
t2 = clip(path=t2_path)
t3 = clip(path=t3_path)

test_path = 'C:\\Users\\cax11\\Desktop\\tmp'
tt_path = os.listdir(test_path)
tt_path = [os.path.join(test_path, f) for f in tt_path]

tt = clip(path=tt_path)
score = mmd(t1, tt)
score2 = mmd(t2, tt)
score3 = mmd(t3, tt)

print(score)
print(score2)
print(score3)
