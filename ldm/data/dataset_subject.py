import json
import cv2
import os
import random
from basicsr.utils import img2tensor
from torch.utils.data import Dataset


class dataset_subject(Dataset):
    def __init__(self, path_json, image_size=512):
        super(dataset_subject, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        self.root_path_im = data['img_pth']
        self.files = data['img_file']
        self.caption = data['caption']
        self.image_size = image_size

    def __getitem__(self, idx):
        name = self.files[idx]
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png', '.jpg')))
        im = cv2.resize(im, (self.image_size, self.image_size))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        return {'im': im, 'sentence': self.caption[idx]}

    def __len__(self):
        return len(self.files)


class dataset_continual(Dataset):
    def __init__(self, path_json, now_task, image_size=512):
        super().__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        self.root_path_im = data['img_pth']
        self.files = data['img_file']
        self.caption = data['caption']
        self.image_size = image_size
        self.now_task = now_task

    def __getitem__(self, idx):
        name = self.files[self.now_task][idx]
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png', 'jpg')))
        im = cv2.resize(im, (self.image_size, self.image_size))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.
        return {'im': im, 'sentence': self.caption[self.now_task][idx]}

    def __len__(self):
        return len(self.files[self.now_task])


class dataset_replay(Dataset):
    def __init__(self, root_path, now_task, iftrain=True, image_size=512):
        super().__init__()
        now_path = os.path.join(root_path, now_task)
        self.iftrain = iftrain
        if iftrain:
            json_path = os.path.join(now_path, f'train{now_task}.json')
        else:
            json_path = os.path.join(now_path, f'val{now_task}.json')
        with open(json_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        if self.iftrain:
            self.root_img_path = [os.path.join(now_path, img_path) for img_path in data['img_file']]
            self.image_size = image_size
        self.caption = data['caption']
        self.rare_token = data['rare_token']

    def get_rare_token(self):
        return self.rare_token

    def __getitem__(self, idx):
        if self.iftrain:
            im = cv2.imread(self.root_img_path[idx])
            im = cv2.resize(im, (self.image_size, self.image_size))
            im = img2tensor(im, bgr2rgb=True, float32=True) / 255.
            return {'im': im, 'sentence': self.caption[idx]}
        else:
            return {'sentence': self.caption[idx]}

    def __len__(self):
        return len(self.caption)


class single_data():
    def __init__(self, data_idx, timesteps):
        self.data_idx = data_idx
        self.total_epoch = 0
        self.data = {}
        self.timesteps = timesteps
        self.step = 10
        assert self.timesteps % self.step == 0, "self.timesteps % self.step != 0"
        idx = list(range(0, self.timesteps, self.step))
        # print(idx)
        self.simple_data = {key: None for key, _ in enumerate(idx)}
        # print(self.simple_data)

    def add_epoch(self):
        self.total_epoch += 1

    def get_simple_data(self):
        while True:
            idx = random.randint(0, self.timesteps / self.step - 1)
            print(idx)
            if self.simple_data[idx] is not None:
                break
        return self.simple_data[idx][1]

    def build_simple_data(self, loss, x):
        t, noise, x_noisy, cond = x
        idx = t // self.step
        if self.simple_data[idx] is None or self.simple_data[idx][0] > loss:
            self.simple_data[idx] = [loss, x]

    def add_data(self, now_epoch, loss, x):
        data = self.data.setdefault(now_epoch, [])
        data.append(x)
        self.build_simple_data(loss, x)

    def get_data(self, now_epoch):
        return self.data[now_epoch]


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    #
    # dataset = dataset_replay(root_path="C:\\Users\\cax11\\Desktop\\replay", now_task='1', iftrain=False, image_size=512)
    # # dataset = dataset_subject(path_json='F:\\dataset\\continual_dog\\dog_train1.json')
    # print(len(dataset))
    # loader = DataLoader(dataset=dataset, batch_size=2)
    # for i, data in enumerate(loader):
    #     # print(i, data['im'])
    #     print(i, data['sentence'])
    # print(type(dataset.get_rare_token()))
    tmp = single_data(data_idx=1, timesteps=1000)
