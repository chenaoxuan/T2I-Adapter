from PIL import Image
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class ClipImageFeature(nn.Module):
    def __init__(self, version="openai/clip-vit-large-patch14", pretrained_model_name_or_path=None, device="cuda",
                 layer="last"):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.model = CLIPModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.device = device
        self.layer = layer
        self.freeze()

    def freeze(self):
        self.transformer = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, path=None, img=None):
        if path is None and img is None:
            raise AttributeError('Both path and tensor are empty')
        if path is not None:
            if isinstance(path, list):
                image = [Image.open(i) for i in path]
            else:
                image = Image.open(path)
            inputs = self.processor(images=image, return_tensors='pt')
        else:
            inputs = self.processor(images=img, return_tensors='pt')
        img_features = self.model.get_image_features(**inputs)

        return img_features


if __name__ == '__main__':
    path1 = "F:\\dataset\\dog\\00.jpg"
    path2 = "F:\\dataset\\dog\\02.jpg"  ##0和2为组1
    path3 = "F:\\dataset\\dog\\01.jpg"
    path4 = "F:\\dataset\\dog\\03.jpg"  ##1和3也为组1
    path5 = "F:\\dataset\\01.jpg"
    path6 = "F:\\dataset\\04.jpg"  ##1和4为组2

    net = ClipImageFeature(pretrained_model_name_or_path="F:\Clip-Vit-Large-Patch14")
    ans1 = net(path=[path1, path2])
    ans2 = net(path=[path3, path4])
    ans3 = net(path=[path5])

    from mmd import mmd

    ss1 = mmd(ans1, ans1)
    ss2 = mmd(ans1, ans3)
    print(ss1, ss2)

    import torch.nn.functional as F

    s1 = F.cosine_similarity(ans1, ans2, dim=1)
    s2 = F.cosine_similarity(ans1, ans3, dim=1)
    print(s1, s2)
