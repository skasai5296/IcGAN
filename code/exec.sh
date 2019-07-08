# env CUDA_VISIBLE_DEVICES=0 nohup python main.py --use_tensorboard --enable_cuda --nz 128 --opt_method Adam --dataset celeba --root_dir ../../../hdd/dsets/CelebA/ --img_dir "cropped" \
# 	--num_epoch 200 --batch_size 128 --image_size 64 --learning_rate 1e-5 > ../logs/celeba_nz128_64_lr1e-5.log &!

# env CUDA_VISIBLE_DEVICES=0 nohup python main.py --use_tensorboard --enable_cuda --nz 128 --opt_method Adam --dataset celeba --root_dir ../../../hdd/dsets/CelebA/ --img_dir "cropped" \
# 	--num_epoch 200 --batch_size 32 --image_size 224 --residual > ../logs/celeba_nz128_224_lr5e-6_res.log &!

# env CUDA_VISIBLE_DEVICES=1 nohup python main.py --batch_size 128 --use_tensorboard --enable_cuda --nz 100 --opt_method Adam --dataset sunattributes --root_dir ../../dsets/SUN_attr/ --image_size 64 > ../logs/sun_nz100_64_lr5e-6.log &!

# env CUDA_VISIBLE_DEVICES=0 nohup python acgan.py --use_tensorboard --enable_cuda --learning_rate 1e-5 --opt_method Adam --nz 100 --dataset celeba --root_dir ../../dsets/CelebA/ --num_epoch 500 --batch_size 256 --image_size 64  > ../logs/cls_celeba_nz100_64_lr1e-5.log &!

# env CUDA_VISIBLE_DEVICES=0 nohup python acgan.py --use_tensorboard --enable_cuda --learning_rate 1e-5 --opt_method Adam --nz 100 --dataset celeba --root_dir ../../../hdd/dsets/CelebA/ --num_epoch 500 --batch_size 256 --image_size 64  > ../logs/cls_celeba_nz100_64_lr1e-5.log &!

# env CUDA_VISIBLE_DEVICES=0 nohup python enc_train.py --use_tensorboard --enable_cuda --learning_rate 1e-3 --opt_method SGD --nz 128 --dataset celeba --root_dir ../../../hdd/dsets/CelebA/ --num_epoch 100 --batch_size 128 --image_size 64 --model_ep 100 --attr_epoch 10 > ../logs/enc_celeba_nz128_64_lr1e-3.log &!

# env CUDA_VISIBLE_DEVICES=0 nohup python enc_train.py --use_tensorboard --enable_cuda --learning_rate 1e-3 --opt_method Adam --nz 128 --dataset celeba \
# 	--root_dir ../../../hdd/dsets/CelebA/ --img_dir "cropped" --num_epoch 100 --batch_size 128 --image_size 64 --model_ep 200 \
# 	--attr_epoch 10 > ../logs/enc_celeba_nz128_64_lr1e-3.log &!

# env CUDA_VISIBLE_DEVICES=0 nohup python langeval.py --use_tensorboard --enable_cuda  --nz 128 --dataset celeba --root_dir ../../../hdd/dsets/CelebA/ --num_epoch 100 --batch_size 128 --image_size 64 --model_ep 100 > ../logs/atteval.log &!
