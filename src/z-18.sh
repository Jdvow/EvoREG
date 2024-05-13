gpu_id=0
dataset='ICEWS18'
hislen=5

python main.py \
 -d ${dataset} \
 --train-history-len ${hislen} \
 --test-history-len ${hislen} \
 --lr 0.001 \
 --n-layers 2 \
 --evaluate-every 1 \
 --n-hidden 200 \
 --self-loop \
 --decoder convtranse \
 --encoder uvrgcn \
 --layer-norm \
 --ft_lr 0.001 \
 --norm_weight 1 \
 --task-weight 0.7 \
 --weight 0.5 \
 --angle 10 \
 --discount 1 \
 --add-static-graph \
 --gpu ${gpu_id} 

 python main.py \
 -d ${dataset} \
 --train-history-len ${hislen} \
 --test-history-len ${hislen} \
 --lr 0.001 \
 --n-layers 2 \
 --evaluate-every 1 \
 --n-hidden 200 \
 --self-loop \
 --decoder convtranse \
 --encoder uvrgcn \
 --layer-norm \
 --ft_lr 0.001 \
 --norm_weight 1 \
 --task-weight 0.7 \
 --weight 0.5 \
 --angle 10 \
 --discount 1 \
 --add-static-graph \
 --gpu ${gpu_id} \
 --test-valid

 python main.py \
 -d ${dataset} \
 --train-history-len ${hislen} \
 --test-history-len ${hislen} \
 --lr 0.001 \
 --n-layers 2 \
 --evaluate-every 1 \
 --n-hidden 200 \
 --self-loop \
 --decoder convtranse \
 --encoder uvrgcn \
 --layer-norm \
 --ft_lr 0.001 \
 --norm_weight 1 \
 --task-weight 0.7 \
 --weight 0.5 \
 --angle 10 \
 --discount 1 \
 --add-static-graph \
 --gpu ${gpu_id} \
 --test-test
