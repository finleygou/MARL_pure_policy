#!/bin/sh
env="MPE"
scenario="simple_formation"
num_landmarks=0
num_agents=3
algo="rmappo"
exp="check"
seed_max=1
use_Relu=False
layer_N=2
hidden_size=32
save_data=False

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs share_policy=1 --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --use_Relu ${use_Relu} --layer_N ${layer_N} --hidden_size ${hidden_size} \
    --n_training_threads 4 --n_rollout_threads 1 --use_render --episode_length 200 --render_episodes 1 \
    --model_dir "/home/sdc/goufandi_space/Projects/SwarmRoundUp_V2/onpolicy/scripts/results/MPE/simple_formation/rmappo/check/wandb/run-20230904_221915-3pc958yn/files/"\
    --save_data ${save_data}
done
# run-20230313_192717-33k2u175 (可以跑但追不上)
# run-20230316_221952-1z7aecet (学偏了)
# run-20230321_001704-1ghex8c5 (best)
# run-20230323_015245-2dkj77wb (best after use all obs)
# run-20230323_113205-1erc1lsu 有围捕动机，但是agent4不收敛
# run-20230323_173010-2w21spc9 最好的曲线但是实验失败
# run-20230325_114944-2n3auo3p BEST of ALL 3.26
# run-20230420_102038-3txi92fv BESTOFALL 4.20