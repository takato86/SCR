source /home/tokudo/Develop/research/venv/bin/activate
start=16
end=20
gpu=1
env_config=configs/env_v1.config
version=v1

cadrl(){
    for i in `seq $start $end`
    do
    python train.py --policy=cadrl --y --env_config=$env_config --output_dir=data/output/cadrl_1human_fixed_${version}_$i --seed=$i --gpu=$gpu &
    done
}

dta(){
    for i in `seq $start $end`
    do
    python train.py --policy=cadrl --shaping=dta --y --env_config=$env_config --output_dir=data/output/cadrl-dtav2_1human_fixed_${version}_$i --seed=$i --gpu=$gpu &
    done
}

nrs(){
    for i in `seq $start $end`
    do
    python train.py --policy=cadrl --shaping=nrs --y --env_config=$env_config --output_dir=data/output/cadrl-nrs_1human_fixed_${version}_$i --seed=$i --gpu=$gpu &
    done
}

repeat(){
    cadrl
    wait
    dta
    wait
    nrs
    wait
}

repeat 
# start=16
# end=20
# repeat
exit 1