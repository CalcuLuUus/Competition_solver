model_path=~/autodl-tmp/gaussian-splatting/output/case1r3

python render.py -m $model_path --skip_train --quiet --eval -i images_gt_downsampled

python metrics.py -m $model_path