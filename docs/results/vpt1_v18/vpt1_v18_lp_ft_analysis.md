# VPT1 v18 LP vs FT Analysis

Source: compiled three-run result tables pasted from the VPT1 v18 run.

## Files

- Linear probe CSV: `docs/results/vpt1_v18/vpt1_v18_linear_probe_results.csv`
- Fine-tune CSV: `docs/results/vpt1_v18/vpt1_v18_finetune_results.csv`
- Pairwise comparison CSV: `docs/results/vpt1_v18/vpt1_v18_lp_vs_ft_comparison.csv`
- ImageNet matched CSV: `docs/results/vpt1_v18/vpt1_v18_lp_ft_imagenet_matched.csv`
- Plotly HTML: `docs/results/vpt1_v18/vpt1_v18_lp_ft_vs_imagenet.html`

## Headline

- LP table: 497 models; total average 55.209%.
- FT table: 494 models; total average 57.294%.
- Shared models: 493.
- Paired mean delta FT-LP: +2.105 percentage points; median +2.265; std 3.016.
- Pearson LP/FT correlation on shared models: 0.301; Spearman rank correlation: 0.236.
- FT beats LP on 379/493 shared models (76.9%).
- ImageNet comparison uses 466 models matched against `results-imagenet.csv`; LP/ImageNet Pearson r=0.429 and FT/ImageNet Pearson r=0.427.

Interpretation: FT worked mechanically, but the gain is modest relative to the training capacity. The weak LP/FT rank correlation means fine-tuning changes which families/models look best; it is not just the LP leaderboard shifted upward.

## Important Caveat

`VPT_code/VPT/run_accel_finetune.py` currently evaluates on the test split each epoch and stores the epoch with best test accuracy. That makes the FT table useful for debugging/model search, but optimistic for thesis reporting. For clean reporting, split train into train/val, select by val, and evaluate once on held-out test.

## Top FT Models

| model | ft_avg | lp_avg | delta |
| --- | ---: | ---: | ---: |
| beitv2_large_patch16_224.in1k_ft_in22k_in1k | 70.664 | 59.961 | 10.703 |
| swinv2_large_window12to24_192to384.ms_in22k_ft_in1k | 70.195 | 57.656 | 12.539 |
| swin_large_patch4_window12_384.ms_in22k_ft_in1k | 69.479 | 56.159 | 13.320 |
| eva_large_patch14_336.in22k_ft_in1k | 69.076 | 57.292 | 11.784 |
| eva02_base_patch14_448.mim_in22k_ft_in22k_in1k | 67.279 | 59.297 | 7.982 |
| beitv2_large_patch16_224.in1k_ft_in1k | 66.693 | 58.906 | 7.787 |
| cait_m36_384.fb_dist_in1k | 66.536 | 56.094 | 10.442 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 66.484 | 59.271 | 7.213 |
| maxvit_large_tf_512.in21k_ft_in1k | 65.937 | 58.581 | 7.356 |
| beit_large_patch16_384.in22k_ft_in22k_in1k | 65.899 | 57.982 | 7.917 |

## Top LP Models

| model | lp_avg | ft_avg | delta |
| --- | ---: | ---: | ---: |
| vit_base_patch14_dinov2.lvd142m | 62.135 | 52.812 | -9.323 |
| vit_large_patch14_reg4_dinov2.lvd142m | 61.940 | 55.143 | -6.797 |
| vit_large_patch14_dinov2.lvd142m | 61.367 | 53.750 | -7.617 |
| vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k | 60.677 | 53.047 | -7.630 |
| convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384 | 60.377 | 62.149 | 1.772 |
| vit_base_patch14_reg4_dinov2.lvd142m | 60.364 | 57.435 | -2.929 |
| convnext_xxlarge.clip_laion2b_soup_ft_in1k | 60.208 | 57.201 | -3.007 |
| beitv2_large_patch16_224.in1k_ft_in22k_in1k | 59.961 | 70.664 | 10.703 |
| convnext_xlarge.fb_in22k_ft_in1k_384 | 59.895 | 63.893 | 3.998 |
| maxvit_large_tf_384.in21k_ft_in1k | 59.766 | 62.760 | 2.994 |

## Biggest FT Gains

| model | ft_avg | lp_avg | delta |
| --- | ---: | ---: | ---: |
| swin_large_patch4_window12_384.ms_in22k_ft_in1k | 69.479 | 56.159 | 13.320 |
| swinv2_large_window12to24_192to384.ms_in22k_ft_in1k | 70.195 | 57.656 | 12.539 |
| eva_large_patch14_336.in22k_ft_in1k | 69.076 | 57.292 | 11.784 |
| beitv2_large_patch16_224.in1k_ft_in22k_in1k | 70.664 | 59.961 | 10.703 |
| cait_m36_384.fb_dist_in1k | 66.536 | 56.094 | 10.442 |
| beitv2_base_patch16_224.in1k_ft_in1k | 64.622 | 55.026 | 9.596 |
| caformer_m36.sail_in1k | 61.393 | 51.940 | 9.453 |
| cait_s36_384.fb_dist_in1k | 64.063 | 54.636 | 9.427 |
| densenet161.tv_in1k | 60.625 | 52.070 | 8.555 |
| dpn68b.mx_in1k | 60.013 | 51.550 | 8.463 |

## Biggest FT Drops

| model | ft_avg | lp_avg | delta |
| --- | ---: | ---: | ---: |
| vit_base_patch14_dinov2.lvd142m | 52.812 | 62.135 | -9.323 |
| vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k | 53.047 | 60.677 | -7.630 |
| vit_large_patch14_dinov2.lvd142m | 53.750 | 61.367 | -7.617 |
| vit_large_patch14_reg4_dinov2.lvd142m | 55.143 | 61.940 | -6.797 |
| vit_base_patch16_clip_224.openai | 52.969 | 58.321 | -5.352 |
| efficientnet_b2.ra_in1k | 51.641 | 56.823 | -5.182 |
| vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b | 53.841 | 58.841 | -5.000 |
| tf_efficientnetv2_b3.in21k_ft_in1k | 52.826 | 57.669 | -4.843 |
| semnasnet_075.rmsp_in1k | 51.614 | 56.042 | -4.428 |
| vit_huge_patch14_clip_224.laion2b_ft_in1k | 54.154 | 58.529 | -4.375 |

## Highest FT Run Variance

| model | ft_avg | ft_run_std | ft_run_range |
| --- | ---: | ---: | ---: |
| eva02_large_patch14_448.mim_in22k_ft_in22k_in1k | 65.612 | 10.360 | 22.734 |
| beitv2_large_patch16_224.in1k_ft_in1k | 66.693 | 8.078 | 19.649 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 66.484 | 7.396 | 17.539 |
| eva_large_patch14_336.in22k_ft_in1k | 69.076 | 6.304 | 15.390 |
| convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384 | 63.216 | 5.988 | 14.102 |
| swinv2_large_window12to16_192to256.ms_in22k_ft_in1k | 63.190 | 5.926 | 12.890 |
| eva_large_patch14_336.in22k_ft_in22k_in1k | 64.375 | 5.675 | 13.515 |
| eva02_large_patch14_448.mim_m38m_ft_in1k | 60.443 | 5.619 | 12.109 |
| swinv2_large_window12to24_192to384.ms_in22k_ft_in1k | 70.195 | 5.186 | 12.656 |
| vit_so400m_patch14_siglip_gap_378.webli_ft_in1k | 61.198 | 5.126 | 12.187 |

## Family Patterns

| family | n | lp_mean | ft_mean | delta_mean | ft_std_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| beitv2 | 4 | 57.712 | 65.788 | +8.076 | 3.823 |
| eva | 4 | 57.552 | 64.531 | +6.979 | 4.719 |
| cait | 5 | 55.195 | 61.286 | +6.091 | 1.401 |
| swin | 10 | 55.177 | 60.249 | +5.072 | 1.235 |
| eva02 | 7 | 58.281 | 62.762 | +4.481 | 4.679 |
| crossvit | 4 | 53.431 | 57.881 | +4.450 | 0.832 |
| repvgg | 10 | 55.292 | 59.409 | +4.117 | 0.940 |
| swinv2 | 9 | 56.058 | 60.137 | +4.080 | 2.971 |
| resmlp | 4 | 53.317 | 57.318 | +4.001 | 1.090 |
| resnest | 4 | 54.085 | 58.037 | +3.952 | 1.154 |
| focalnet | 6 | 54.164 | 58.108 | +3.943 | 1.459 |
| beit | 5 | 58.091 | 61.985 | +3.893 | 1.968 |
| convnextv2 | 8 | 56.299 | 60.132 | +3.833 | 0.865 |
| regnetx | 8 | 53.630 | 57.391 | +3.761 | 0.710 |
| maxvit | 13 | 55.655 | 59.266 | +3.611 | 1.071 |
| poolformerv2 | 4 | 54.759 | 58.268 | +3.509 | 1.251 |
| twins | 6 | 53.902 | 57.405 | +3.503 | 1.261 |
| res2net50 | 5 | 54.266 | 57.716 | +3.451 | 0.938 |
| hrnet | 9 | 56.016 | 59.255 | +3.239 | 0.580 |
| mvitv2 | 4 | 55.251 | 58.349 | +3.099 | 1.151 |
| deit3 | 8 | 55.656 | 58.722 | +3.067 | 1.247 |
| caformer | 6 | 55.862 | 58.878 | +3.017 | 1.129 |
| volo | 7 | 54.280 | 57.206 | +2.926 | 1.114 |
| deit | 7 | 54.094 | 56.834 | +2.740 | 1.229 |

## Input Size Pattern

| inferred_size | n | lp_mean | ft_mean | delta_mean |
| --- | ---: | ---: | ---: | ---: |
| 384 | 31 | 56.999 | 60.208 | +3.209 |
| 224 | 130 | 54.970 | 57.144 | +2.174 |
| unknown | 301 | 54.958 | 56.825 | +1.867 |

## Model Set Mismatch

- Only in LP: 4 models.
- Only in FT: 1 models.
- LP-only examples: cait_m48_448.fb_dist_in1k, convnextv2_huge.fcmae_ft_in22k_in1k_512, eva_giant_patch14_560.m30m_ft_in22k_in1k, maxvit_xlarge_tf_384.in21k_ft_in1k.
- FT-only examples: vit_large_patch16_siglip_384.webli.
