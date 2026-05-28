# VPT Thesis Results and Plan

## Status Snapshot

This note summarizes the current VPT1/VPT1-depth evidence and the expected next
experiments. It is repo-local handoff context for future Codex/Claude work.

## VPT1 v18 RGB Results

Files:

- `docs/results/vpt1_v18/vpt1_v18_linear_probe_results.csv`
- `docs/results/vpt1_v18/vpt1_v18_finetune_results.csv`
- `docs/results/vpt1_v18/vpt1_v18_lp_vs_ft_comparison.csv`
- `docs/results/vpt1_v18/vpt1_v18_lp_ft_analysis.md`

Headline:

- Linear probe: 497 model rows, mean 55.209%, median 54.922%, best 62.135%.
- Fine-tune: 494 model rows, mean 57.294%, median 57.083%, best 70.664%.
- Shared models: 493.
- Mean paired FT-LP delta: +2.105 percentage points.
- FT beats LP on 379/493 shared models.
- LP/ImageNet and FT/ImageNet Pearson correlations are similar, about 0.43.

Interpretation:

- VPT1 RGB is learnable but modest in absolute accuracy.
- Fine-tuning improves average performance, but changes model ranking; the LP
  leaderboard is not a reliable proxy for FT ranking.
- Treat current FT as debugging/model-search evidence because the current FT
  script selected best epoch by test accuracy.

Top RGB LP models:

| Model | Avg Acc |
|---|---:|
| `vit_base_patch14_dinov2.lvd142m` | 62.135 |
| `vit_large_patch14_reg4_dinov2.lvd142m` | 61.940 |
| `vit_large_patch14_dinov2.lvd142m` | 61.367 |
| `maxvit_xlarge_tf_384.in21k_ft_in1k` | 61.276 |
| `vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k` | 60.677 |

Top RGB FT models:

| Model | Avg Acc |
|---|---:|
| `beitv2_large_patch16_224.in1k_ft_in22k_in1k` | 70.664 |
| `swinv2_large_window12to24_192to384.ms_in22k_ft_in1k` | 70.195 |
| `swin_large_patch4_window12_384.ms_in22k_ft_in1k` | 69.479 |
| `eva_large_patch14_336.in22k_ft_in1k` | 69.076 |
| `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | 67.279 |

## VPT1 v18 Depth Results

File:

- `docs/results/vpt1_v18_depth/vpt1_v18_depth_linear_probe_results.csv`

Headline:

- Depth linear probe: 498 model rows, mean 78.210%, median 78.209%, best 85.521%.
- Shared with RGB LP: 497 models.
- Depth LP beats RGB LP on all 497 shared models.
- Mean depth-RGB LP delta: +23.004 percentage points; median +23.086.

Top depth LP models:

| Model | Avg Acc |
|---|---:|
| `tf_efficientnet_b7.ap_in1k` | 85.521 |
| `regnety_320.swag_ft_in1k` | 84.727 |
| `cait_m36_384.fb_dist_in1k` | 84.075 |
| `vit_large_patch16_384.augreg_in21k_ft_in1k` | 84.036 |
| `convnextv2_huge.fcmae_ft_in22k_in1k_512` | 84.010 |
| `swin_large_patch4_window12_384.ms_in22k_ft_in1k` | 83.932 |
| `eca_nfnet_l1.ra2_in1k` | 83.906 |
| `convnextv2_large.fcmae_ft_in22k_in1k_384` | 83.750 |
| `convnext_xxlarge.clip_laion2b_soup_ft_in1k` | 83.373 |
| `repvgg_b1g4.rvgg_in1k` | 83.372 |

Interpretation:

- Depth makes VPT1 much easier for first-frame linear readout than RGB.
- Depth FT is expected to be the next important check: if it follows the RGB
  pattern, it should improve over depth LP, but the current FT protocol must be
  treated carefully until validation-selected checkpointing exists.

## Expected Next Experiments

VPT1 depth:

- Analyze depth FT when the active jobs finish.
- Compare depth FT to depth LP, RGB FT, and ImageNet top-1.
- Use the depth result to decide whether depth is a strong thesis baseline or a
  sanity-check/control condition.

VPT2:

- Build the VPT2 dataset after active generation completes.
- Run LP + FT on the same model suite.
- Compare VPT2 against VPT1 RGB and VPT1 depth to measure task transfer and
  difficulty shift.

Strategy datasets:

- Generate VPT1 Strategy dataset.
- Generate VPT2 Strategy dataset.
- Build/validate both datasets before training.
- Run the same LP and FT model suites on VPT1 Strategy and VPT2 Strategy.
- Compare strategy performance against normal VPT1/VPT2 to test whether
  strategy-enriched generation improves downstream model readout.

## Reporting Caveat

Do not present the existing FT tables as final thesis-grade numbers until the
FT protocol selects checkpoints on validation accuracy and evaluates once on a
held-out test set. Current FT tables are still useful for debugging, ranking
families, and estimating whether FT moves the signal.

