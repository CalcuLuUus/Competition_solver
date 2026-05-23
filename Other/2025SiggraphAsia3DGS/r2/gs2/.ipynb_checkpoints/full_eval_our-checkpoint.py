#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
import time


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--use_depth", action="store_true")
parser.add_argument("--use_expcomp", action="store_true")
parser.add_argument("--fast", action="store_true")
parser.add_argument("--aa", action="store_true")


args, _ = parser.parse_known_args()

all_scenes = []

if not args.skip_training or not args.skip_rendering or not args.skip_metrics:
    parser.add_argument("--ourdata", "-od", required=True, type=str)
    args = parser.parse_args()
    our_data_scenes = os.listdir(args.ourdata)
    all_scenes.extend(our_data_scenes)
    
if not args.skip_training:
    common_args = " --disable_viewer --quiet --eval --test_iterations -1 --iterations 6000 --densify_until_iter 5400 --save_iterations 6000 --densify_mode freq --resolution_mode freq  --max_n_gaussian -1"
    
    if args.aa:
        common_args += " --antialiasing "
    if args.use_depth:
        common_args += " -d depths2/ "

    if args.use_expcomp:
        common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 "

    if args.fast:
        common_args += " --optimizer_type sparse_adam "

    start_time = time.time()
    
    for scene in our_data_scenes:
        source = args.ourdata + "/" + scene
        os.system("python train_dash.py -i images_gt_downsampled -s " + source + "  -m " + args.output_path + "/" + scene + common_args)
    db_timing = (time.time() - start_time)/60.0

    with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
        file.write(f"time: {db_timing} minutes for {len(our_data_scenes)} scenes\n")

if not args.skip_rendering:
    all_sources = []

    for scene in our_data_scenes:
        all_sources.append(args.ourdata + "/" + scene)
    
    common_args = " --quiet --eval --skip_train -i images_gt_downsampled"
    
    if args.aa:
        common_args += " --antialiasing "
    if args.use_expcomp:
        common_args += "  "

    for scene, source in zip(all_scenes, all_sources):
        # os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        # os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py  -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string + " >> log.txt")
