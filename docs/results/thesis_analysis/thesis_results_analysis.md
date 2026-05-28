# VPT Thesis Results Analysis

Source tables are the compiled three-run LP/FT result CSVs saved under `docs/results/`.
ImageNet top-1 and parameter counts come from `results-imagenet.csv`.

## Output Files

- `all_task_matched_results.csv`: all matched model/task rows with ImageNet and parameter metadata.
- `plots/all_tasks_imagenet_top1_vs_accuracy.{html,png}`
- `plots/all_tasks_param_count_vs_accuracy.{html,png}`
- Per-task matched CSVs and per-task plots under each task/plot filename.

## Headline Totals

| task | LP total avg | FT total avg | FT-LP | LP models | FT models | ImageNet matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VPT1 v18 | 55.209 | 57.294 | +2.085 | 497 | 494 | 466 |
| VPT1 v18 Depth | 78.210 | 88.934 | +10.724 | 498 | 494 | 466 |
| VPT2 v4 | 51.918 | 86.703 | +34.785 | 498 | 494 | 466 |

## Correlations

| task | metric | ImageNet Pearson | ImageNet Spearman | params Pearson | params Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| VPT1 v18 | LP | 0.429 | 0.423 | 0.546 | 0.414 |
| VPT1 v18 | FT | 0.427 | 0.391 | 0.397 | 0.558 |
| VPT1 v18 Depth | LP | 0.441 | 0.413 | 0.346 | 0.436 |
| VPT1 v18 Depth | FT | 0.561 | 0.712 | 0.372 | 0.671 |
| VPT2 v4 | LP | -0.093 | 0.002 | 0.142 | 0.181 |
| VPT2 v4 | FT | 0.241 | 0.189 | 0.179 | 0.425 |

## Top Models By Task

### VPT1 v18

Top FT:

| model | FT | LP | FT-LP | ImageNet | params M | FT range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beitv2_large_patch16_224.in1k_ft_in22k_in1k | 70.664 | 59.961 | 10.703 | 88.406 | 304.430 | 7.617 |
| swinv2_large_window12to24_192to384.ms_in22k_ft_in1k | 70.195 | 57.656 | 12.539 | 87.474 | 196.740 | 12.656 |
| swin_large_patch4_window12_384.ms_in22k_ft_in1k | 69.479 | 56.159 | 13.320 | 87.142 | 196.740 | 2.187 |
| eva_large_patch14_336.in22k_ft_in1k | 69.076 | 57.292 | 11.784 | 88.680 | 304.530 | 15.390 |
| eva02_base_patch14_448.mim_in22k_ft_in22k_in1k | 67.279 | 59.297 | 7.982 | 88.678 | 87.120 | 6.132 |
| beitv2_large_patch16_224.in1k_ft_in1k | 66.693 | 58.906 | 7.787 | 87.414 | 304.430 | 19.649 |
| cait_m36_384.fb_dist_in1k | 66.536 | 56.094 | 10.442 | 86.060 | 271.220 | 4.219 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 66.484 | 59.271 | 7.213 | 90.056 | 305.080 | 17.539 |
| maxvit_large_tf_512.in21k_ft_in1k | 65.937 | 58.581 | 7.356 | 88.236 | 212.330 | 4.141 |
| beit_large_patch16_384.in22k_ft_in22k_in1k | 65.899 | 57.982 | 7.917 | 88.380 | 305.000 | 5.000 |

Top LP:

| model | LP | FT | FT-LP | ImageNet | params M |
| --- | ---: | ---: | ---: | ---: | ---: |
| vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k | 60.677 | 53.047 | -7.630 | 88.634 | 632.460 |
| convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384 | 60.377 | 62.149 | 1.772 | 88.334 | 200.130 |
| convnext_xxlarge.clip_laion2b_soup_ft_in1k | 60.208 | 57.201 | -3.007 | 88.622 | 846.470 |
| beitv2_large_patch16_224.in1k_ft_in22k_in1k | 59.961 | 70.664 | 10.703 | 88.406 | 304.430 |
| convnext_xlarge.fb_in22k_ft_in1k_384 | 59.895 | 63.893 | 3.998 | 87.764 | 350.200 |
| maxvit_large_tf_384.in21k_ft_in1k | 59.766 | 62.760 | 2.994 | 87.994 | 212.030 |
| swinv2_base_window12to24_192to384.ms_in22k_ft_in1k | 59.753 | 61.341 | 1.588 | 87.142 | 87.920 |
| convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384 | 59.505 | 63.216 | 3.711 | 87.144 | 88.590 |
| beit_large_patch16_512.in22k_ft_in22k_in1k | 59.323 | 59.388 | 0.065 | 88.576 | 305.670 |
| eva02_base_patch14_448.mim_in22k_ft_in22k_in1k | 59.297 | 67.279 | 7.982 | 88.678 | 87.120 |

Largest FT run instability:

| model | FT | FT range | LP |
| --- | ---: | ---: | ---: |
| eva02_large_patch14_448.mim_in22k_ft_in22k_in1k | 65.612 | 22.734 | 58.425 |
| beitv2_large_patch16_224.in1k_ft_in1k | 66.693 | 19.649 | 58.906 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 66.484 | 17.539 | 59.271 |
| eva_large_patch14_336.in22k_ft_in1k | 69.076 | 15.390 | 57.292 |
| convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384 | 63.216 | 14.102 | 59.505 |
| eva_large_patch14_336.in22k_ft_in22k_in1k | 64.375 | 13.515 | 58.802 |
| swinv2_large_window12to16_192to256.ms_in22k_ft_in1k | 63.190 | 12.890 | 56.901 |
| swinv2_large_window12to24_192to384.ms_in22k_ft_in1k | 70.195 | 12.656 | 57.656 |

### VPT1 v18 Depth

Top FT:

| model | FT | LP | FT-LP | ImageNet | params M | FT range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eva02_large_patch14_448.mim_m38m_ft_in1k | 95.182 | 78.789 | 16.393 | 89.550 | 305.080 | 0.234 |
| eva02_large_patch14_448.mim_in22k_ft_in22k_in1k | 94.922 | 82.227 | 12.695 | 89.956 | 305.080 | 0.351 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 94.687 | 81.836 | 12.851 | 90.056 | 305.080 | 0.781 |
| eva_large_patch14_336.in22k_ft_in22k_in1k | 94.675 | 81.771 | 12.904 | 89.238 | 304.530 | 0.391 |
| eva02_large_patch14_448.mim_in22k_ft_in1k | 94.505 | 78.503 | 16.002 | 89.634 | 305.080 | 0.820 |
| eva02_base_patch14_448.mim_in22k_ft_in1k | 94.349 | 75.716 | 18.633 | 88.262 | 87.120 | 0.273 |
| eva02_base_patch14_448.mim_in22k_ft_in22k_in1k | 94.349 | 80.365 | 13.984 | 88.678 | 87.120 | 1.133 |
| maxvit_large_tf_384.in1k | 94.258 | 78.399 | 15.859 | 86.242 | 212.030 | 0.234 |
| maxvit_tiny_tf_512.in1k | 94.036 | 70.820 | 23.216 | 85.658 | 31.050 | 0.508 |
| volo_d2_384.sail_in1k | 94.036 | 77.110 | 16.926 | 86.054 | 58.870 | 0.469 |

Top LP:

| model | LP | FT | FT-LP | ImageNet | params M |
| --- | ---: | ---: | ---: | ---: | ---: |
| tf_efficientnet_b7.ap_in1k | 85.521 | 88.958 | 3.437 | 85.132 | 66.350 |
| regnety_320.swag_ft_in1k | 84.727 | 91.940 | 7.213 | 86.830 | 145.050 |
| cait_m36_384.fb_dist_in1k | 84.075 | 93.451 | 9.376 | 86.060 | 271.220 |
| vit_large_patch16_384.augreg_in21k_ft_in1k | 84.036 | 92.825 | 8.789 | 87.096 | 304.720 |
| swin_large_patch4_window12_384.ms_in22k_ft_in1k | 83.932 | 93.073 | 9.141 | 87.142 | 196.740 |
| eca_nfnet_l1.ra2_in1k | 83.906 | 91.159 | 7.253 | 83.264 | 41.410 |
| convnextv2_large.fcmae_ft_in22k_in1k_384 | 83.750 | 92.643 | 8.893 | 88.180 | 197.960 |
| convnext_xxlarge.clip_laion2b_soup_ft_in1k | 83.373 | 92.995 | 9.622 | 88.622 | 846.470 |
| repvgg_b1g4.rvgg_in1k | 83.372 | 89.076 | 5.704 | 77.608 | 39.970 |
| swinv2_large_window12to24_192to384.ms_in22k_ft_in1k | 83.320 | 93.268 | 9.948 | 87.474 | 196.740 |

Largest FT run instability:

| model | FT | FT range | LP |
| --- | ---: | ---: | ---: |
| vit_base_patch16_clip_224.openai_ft_in1k | 85.000 | 20.625 | 76.276 |
| convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384 | 87.305 | 17.109 | 81.601 |
| vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k | 88.932 | 13.477 | 81.003 |
| mobilenetv2_050.lamb_in1k | 70.000 | 5.273 | 76.120 |
| dla46x_c.in1k | 79.076 | 4.961 | 74.922 |
| shvit_s4.in1k | 80.547 | 4.805 | 76.406 |
| spnasnet_100.rmsp_in1k | 75.443 | 4.219 | 78.451 |
| repvit_m1_0.dist_450e_in1k | 73.984 | 4.063 | 75.664 |

### VPT2 v4

Top FT:

| model | FT | LP | FT-LP | ImageNet | params M | FT range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vgg19.tv_in1k | 99.662 | 57.956 | 41.706 | 72.400 | 143.670 | 0.274 |
| vgg19_bn.tv_in1k | 99.648 | 59.739 | 39.909 | 74.230 | 143.680 | 0.000 |
| vgg16_bn.tv_in1k | 99.609 | 57.813 | 41.796 | 73.354 | 138.370 | 0.078 |
| vgg11.tv_in1k | 99.557 | 63.034 | 36.523 | 69.050 | 132.860 | 0.039 |
| vgg16.tv_in1k | 99.557 | 63.828 | 35.729 | 71.594 | 138.360 | 0.078 |
| resmlp_36_224.fb_in1k | 99.453 | 52.201 | 47.252 | 79.772 | 44.690 | 0.273 |
| vgg13.tv_in1k | 99.440 | 59.414 | 40.026 | 69.950 | 133.050 | 0.156 |
| mixer_b16_224.goog_in21k_ft_in1k | 99.414 | 52.383 | 47.031 | 76.616 | 59.880 | 0.117 |
| eva_large_patch14_196.in22k_ft_in22k_in1k | 99.388 | 51.979 | 47.409 | 88.590 | 304.140 | 0.390 |
| resmlp_12_224.fb_in1k | 99.388 | 52.682 | 46.706 | 76.660 | 15.350 | 0.429 |

Top LP:

| model | LP | FT | FT-LP | ImageNet | params M |
| --- | ---: | ---: | ---: | ---: | ---: |
| vgg16.tv_in1k | 63.828 | 99.557 | 35.729 | 71.594 | 138.360 |
| vgg11.tv_in1k | 63.034 | 99.557 | 36.523 | 69.050 | 132.860 |
| vgg13_bn.tv_in1k | 62.318 | 99.075 | 36.757 | 71.560 | 133.050 |
| resnet50d.a3_in1k | 59.961 | 91.667 | 31.706 | 77.222 | 25.580 |
| vgg19_bn.tv_in1k | 59.739 | 99.648 | 39.909 | 74.230 | 143.680 |
| vgg13.tv_in1k | 59.414 | 99.440 | 40.026 | 69.950 | 133.050 |
| resnet200d.ra2_in1k | 58.568 | 97.956 | 39.388 | 83.250 | 64.690 |
| vgg11_bn.tv_in1k | 58.255 | 99.271 | 41.016 | 70.374 | 132.870 |
| seresnet50.a3_in1k | 57.956 | 80.547 | 22.591 | 75.104 | 28.090 |
| vgg19.tv_in1k | 57.956 | 99.662 | 41.706 | 72.400 | 143.670 |

Largest FT run instability:

| model | FT | FT range | LP |
| --- | ---: | ---: | ---: |
| vit_base_patch8_224.augreg2_in21k_ft_in1k | 82.669 | 48.476 | 50.938 |
| vit_large_patch14_clip_224.openai_ft_in12k_in1k | 67.982 | 47.578 | 51.172 |
| maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k | 82.083 | 47.500 | 52.930 |
| vit_large_patch16_384.augreg_in21k_ft_in1k | 81.536 | 47.343 | 51.211 |
| vit_so400m_patch14_siglip_gap_378.webli_ft_in1k | 83.919 | 46.406 | 51.471 |
| vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k | 72.370 | 45.117 | 53.737 |
| vit_large_patch14_clip_224.openai_ft_in1k | 82.578 | 45.079 | 52.031 |
| vit_base_patch16_clip_224.laion2b_ft_in12k_in1k | 69.154 | 45.039 | 49.922 |

## Cross-Task Notes

- VPT1 v18 LP is near the high-50s and FT only modestly improves it, so the normal RGB task remains difficult under this setup.
- VPT1 depth LP is already strong, and FT jumps sharply; this is the cleanest current task separation.
- VPT2 LP is near chance, but FT jumps high. That gap is useful but should be treated carefully because many VPT2 FT models show large run-to-run instability.
- Parameter count is not a reliable standalone predictor here; check the parameter scatter and the correlation table before using size as a model-selection rule.
- ImageNet top-1 is most useful as a weak prior, not as a direct thesis-task predictor.

## Caveat

If the FT training code still selects checkpoints by test accuracy, the FT numbers are optimistic for final thesis reporting. Use these for exploration/model selection, then rerun finalists with train/val selection and one held-out test evaluation.
