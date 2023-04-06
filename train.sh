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