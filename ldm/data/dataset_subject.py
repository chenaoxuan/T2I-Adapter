import json
import cv2
import os
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
        return {'im': im, 'sentence': self.caption[self.now_task]}

    def __len__(self):
        return len(self.files[self.now_task])


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = dataset_continual(path_json='F:\\dataset\\continual_dog\\dog_train.json', now_task='3')
    # dataset = dataset_subject(path_json='F:\\dataset\\continual_dog\\dog_train1.json')
    print(len(dataset))
    loader = DataLoader(dataset=dataset, batch_size=2)
    for i, data in enumerate(loader):
        # print(i, data['im'])
        print(i, data['sentence'])
