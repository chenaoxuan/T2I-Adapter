from ldm.modules.encoders.adapter import Adapter
from ldm.data.dataset_subject import dataset_subject
import torch

path_json_train = 'F:\\dataset\\dreambooth\\dog\\dog_train.json'
train_dataset = dataset_subject(
    path_json=path_json_train,
    image_size=512
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=0,
    pin_memory=True)

model_ad = Adapter(cin=int(3 * 64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)

for _, data in enumerate(train_dataloader):
    features = model_ad(data['im'])
    print(type(features))
    print(len(features))
    for feature in features:
        print(feature.shape)
    break
