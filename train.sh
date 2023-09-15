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


python tmp.py --ckpt /hy-tmp/sd-v1-4.ckpt --gpu --epochs 3 --print_fq 25 --n_samples 1
python train_singleadapter.py --ckpt /hy-tmp/sd-v1-4.ckpt --gpu --epochs 500 --print_fq 40 --n_samples 4
python tmp.py --ckpt /hy-tmp/sd-v1-4.ckpt --gpu --epochs 500 --print_fq 40 --n_samples 4


oss login
oss cp oss://Clip-Vit-Large-Patch14.zip /hy-tmp
#oss cp oss://Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese.zip /hy-tmp
oss cp oss://sd-v1-4.zip /hy-tmp
#oss cp oss://all.zip /hy-tmp
oss cp oss://replay.zip /hy-tmp
oss cp oss://dataset.zip /hy-tmp


unzip -d Clip-Vit-Large-Patch14 Clip-Vit-Large-Patch14.zip
unzip sd-v1-4.zip
#unzip -d all all.zip
unzip replay.zip
unzip T2I-Adapter-main.zip
unzip -d dataset dataset.zip

cd hy-tmp
cd hy-tmp/T2I-Adapter


zip -r train_subj.zip train_subj/
zip -r data_replay.zip train_subj/
zip -r samples.zip samples/



Dreambooth
python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --skip_grid --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt /hy-tmp/sd-v1-4.ckpt --prompt "a photo of a dog"

python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume /hy-tmp/sd-v1-4.ckpt -n dog --gpus 0, --data_root /hy-tmp/dataset/my/10 --reg_data_root /hy-tmp/dataset/reg --class_word dog


python stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 1 --skip_grid --scale 10.0 --ddim_steps 100 --ckpt /hy-tmp/last.ckpt --prompt "a waj dog on the beach"