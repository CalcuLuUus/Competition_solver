python train.py -s ~/autodl-tmp/testdataset/1747834320424 -m output/case1r2 \
-i images_gt --disable_viewer --eval \
--iterations 7000 --densify_until_iter 4000 \
--test_iterations 7000 \
--save_iterations 7000 \
--optimizer_type sparse_adam \
# --densify_mode freq --resolution_mode freq  --max_n_gaussian -1


# for case in /root/autodl-tmp/Finalconvert/*; do
#   if [ -d "$case" ]; then
#     case=$(basename "$case")
#     echo $case

#     python train.py -s ~/autodl-tmp/Finalconvert/$case -m output/$case \
#     -r 4 \
#     --iterations 10000 \
#     --test_iterations 5000 10000 \
#     --save_iterations 10000 \
#     --eval --port 6006 --optimizer_type sparse_adam

#     model_path=~/autodl-tmp/gaussian-splatting/output/$case

#     python render.py -m $model_path  -r 4

#     python metrics.py -m $model_path >> baseline_metrics.txt
#   fi
# done