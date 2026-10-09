# Upstream sources and attribution

This package uses **RAEv2: Improved Baselines with Representation Autoencoders**,
by Jaskirat Singh, Boyang Zheng, Zongze Wu, Richard Zhang, Eli Shechtman and
Saining Xie. Upstream: https://github.com/nanovisionx/RAEv2

Pinned source: `8a0d238f8dc3b261aba98b217f6c79c0182e8e94`.
RAEv2 is licensed under **CC BY-NC 4.0** (https://creativecommons.org/licenses/by-nc/4.0/); consult its LICENSE for the full terms.
`rae_imagenet/model.py` adapts the upstream DDT forward pass to enable activation
checkpointing and retains this attribution. The rest of the upstream model,
encoder and decoder code is imported from a separate user-installed checkout;
it is not copied into this release.

DINOv3 source: https://github.com/facebookresearch/dinov3
Pinned source: `94a96ac83c2446f15f9bdcfae23cad3c6a9d4988`.
DINOv3 weights/code have their own upstream terms. Dataset and model weights are
not included in this repository. See each provider's terms when obtaining them.

This experiment uses the RAE representation and DDT architecture with a custom
standard-normal, label-conditioned, direct-velocity FM recipe. It is not a
reproduction of the official RAEv2 x-prediction/logit-normal/IG/GMuon recipe.
