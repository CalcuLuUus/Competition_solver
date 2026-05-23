python train_dash.py -s ~/autodl-tmp/testdataset/1750824904001 -m output/case1r4 \
-i images_gt_downsampled --disable_viewer --eval \
--iterations 6000 --densify_until_iter 5400 \
--test_iterations 6000 \
--save_iterations 6000 \
--optimizer_type sparse_adam \
--densify_mode freq --resolution_mode freq  --max_n_gaussian -1 --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp