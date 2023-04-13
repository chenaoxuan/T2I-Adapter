from ldm.data.dataset_subject import dataset_subject
import torch

path_json_train = 'F:/dataset/dreambooth/dog/dog_train.json'
train_dataset = dataset_subject(
    path_json=path_json_train,
    image_size=512
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=0,
    pin_memory=True)

for _, data in enumerate(train_dataloader):
    print(data['sentence'])
    break
