source /home/tokudo/Develop/research/venv/bin/activate
start=10
end=15

cadrl(){
    for i in `seq $start $end`
    do
    python train.py --policy=cadrl --y --output_dir=data/output/cadrl_1human_fixed_$i --seed=$i &
    done
}

dta(){
    for i in `seq $start $end`
    do
    python train.py --policy=cadrl --shaping=dta --y --output_dir=data/output/cadrl-dtav2_1human_fixed_$i --seed=$i &
    done
}

nrs(){
    for i in `seq $start $end`
    do
    python train.py --policy=cadrl --shaping=nrs --y --output_dir=data/output/cadrl-nrs_1human_fixed_$i --seed=$i &
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
start=16
end=20
repeat
exit 1