epoch=$1
model_name=$2
protocol=$3
batch_size=$4

module load opencv/3.4.1
export PATH=/usr/sbin/:$PATH
source ~/.bashrc
conda activate tf_test

mkdir -p /data/stars/user/sdas/NTU_RGB/patches_full_body
sudo mountimg  /data/stars/user/sdas/NTU_RGB/patches_full_body.squashfs /data/stars/user/sdas/NTU_RGB/patches_full_body/
mkdir -p weights_${model_name}

time python train_ST_gcnn_attention.py $epoch $model_name $protocol $batch_size
echo 'Successfully terminated'
