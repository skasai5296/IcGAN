# env CUDA_VISIBLE_DEVICES=0 nohup python main.py --use_tensorboard --enable_cuda --nz 256 --dataset celeba --root_dir ../../dsets/CelebA/ --num_epoch 500 --batch_size 512 --image_size 64  > ../logs/celeba_nz256_64_lr5e-6.log &!
# env CUDA_VISIBLE_DEVICES=1 nohup python main.py --batch_size 128 --use_tensorboard --enable_cuda --nz 100 --dataset sunattributes --root_dir ../../dsets/SUN_attr/ --image_size 64 > ../logs/sun_nz100_64_lr5e-6.log &!


env CUDA_VISIBLE_DEVICES=1 nohup python acgan.py --use_tensorboard --enable_cuda --opt_method Adam --nz 100 --dataset celeba --root_dir ../../dsets/CelebA/ --num_epoch 500 --batch_size 512 --image_size 64  > ../logs/cls_celeba_nz100_64_lr5e-6.log &!
