# set your gpu id
gpus=0
# number of gpus
gpun=1
master_addr=127.0.0.1
master_port=5678

CUDA_VISIBLE_DEVICES=$gpus

python -m torch.distributed.launch --nproc_per_node=$gpun \
  --master_addr=$master_addr --master_port=$master_port train_subj.py --ckpt F:\\sd-v1-4.ckpt
#
#python -m torch.distributed.launch --nproc_per_node=$gpun train_subj.py --ckpt F:\\sd-v1-4.ckpt


python train_subj.py --ckpt F:\\sd-v1-4.ckpt --gpu --epochs 500 --print_fq 25

python train_subj.py --ckpt /hy-tmp/sd-v1-4.ckpt --gpu --epochs 1000 --print_fq 25 --n_samples 2


python tmp.py --ckpt /hy-tmp/sd-v1-4.ckpt --gpu --epochs 500 --print_fq 25 --n_samples 4
python tmp.py --ckpt /hy-tmp/sd-v1-4.ckpt --gpu --epochs 300 --print_fq 25 --n_samples 6


oss cp oss://Clip-Vit-Large-Patch14.zip /hy-tmp
oss cp oss://Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese.zip /hy-tmp
oss cp oss://sd-v1-4.zip /hy-tmp
oss cp oss://all.zip /hy-tmp
oss cp oss://continual_dog.zip /hy-tmp

unzip -d Clip-Vit-Large-Patch14 Clip-Vit-Large-Patch14.zip
unzip sd-v1-4.zip
unzip -d all all.zip
unzip -d continual_dog continual_dog.zip

cd hy-tmp
cd hy-tmp/T2I-Adapter


zip -r train_subj.zip train_subj/
zip -r one.zip train_subj/