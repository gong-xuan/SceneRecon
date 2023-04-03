CUDA_VISBILE_DEVICES='2,3' python generate_tsdf.py --cfg colonrecon --gpu 3 --voxel_size 1 --max_depth 110 --min_distance 1 --min_angle 2 --n_proc 8 --n_gpu 1


python generate_tsdf.py --cfg colonrecon --gpu 3 --voxel_size 1 --max_depth 110 --min_distance 1 --min_angle 2