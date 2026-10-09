"""Adapter around upstream RAEv2 DDT; upstream files are not vendored here.

Checkpointed forward follows nanovisionx/RAEv2's DiTwDDTHead.forward (CC BY-NC
4.0; see THIRD_PARTY.md). We retain the 4 time / 8 class tokens, omit IG, and
predict velocity directly. Conditional dropout belongs to the FM objective.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.checkpoint import checkpoint


def upstream_path(rae_repo):
    src = Path(rae_repo).resolve() / 'src'
    if not (src / 'stage2/models/DDT.py').is_file():
        raise FileNotFoundError(f'RAEv2 source missing: {src}; run rae_imagenet.setup_vendor')
    if str(src) not in sys.path:
        sys.path.append(str(src))


def build_model(config, rae_repo, checkpointing=True):
    upstream_path(rae_repo)
    from stage2.models.DDT import DiTwDDTHead

    class VelocityDDT(DiTwDDTHead):
        def forward(self, x, t, labels):
            if not (checkpointing and self.training and torch.is_grad_enabled()):
                return super().forward(x, t, context=labels)
            seq, time_base = self._build_sequence(x, t, {'context': labels})
            mask = self._build_attn_mask(seq, {})
            for i in range(self.num_enc_blocks):
                seq = checkpoint(self.blocks[i], seq, self.enc_rope, mask, use_reentrant=False)
            seq = self.s_projector(torch.nn.functional.silu(
                time_base + seq[:, :self.s_embedder.num_patches]))
            x = self.x_embedder(x)
            for i in range(self.num_dec_blocks):
                x = checkpoint(self.blocks[self.num_enc_blocks + i], x, seq,
                               self.dec_rope, use_reentrant=False)
            return self.unpatchify(self.final_layer(x, seq), self.x_patch_size)

    return VelocityDDT(**config, condition_type='label',
                       cond_arch=SimpleNamespace(num_t_tokens=4, num_c_tokens=8))
