"""
Ablation hooks for LLaMA-architecture models (Hugging Face `LlamaForCausalLM`).

Every ablation is a forward (pre-)hook, so the weights are never modified and
any combination of ablations can be switched on/off between evaluations:

  heads        zero individual attention heads. A pre-hook on o_proj multiplies
               the concatenated head outputs by a 0/1 mask, so a zeroed head
               contributes nothing to the residual stream.
  attn_layers  zero the whole attention block of a layer (all heads masked).
  mlp_layers   zero the MLP (feed-forward) output of a layer.
  skip_layers  skip a decoder layer entirely (its output := its input).
  embed_noise  add Gaussian noise to the input embeddings, std = sigma * RMS(embedding).
  embed_mask   zero the input embedding of a random fraction of tokens
               (BOS is never masked).
"""
import contextlib
import torch


class Ablator:
    def __init__(self, model):
        self.model = model
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.layers = model.model.layers
        dev = next(model.parameters()).device
        # head mask (float so it can also carry gradients for importance scores)
        self.head_mask = torch.ones(self.n_layers, self.n_heads, device=dev)
        self.mlp_off, self.skip = set(), set()
        self.embed_noise, self.embed_mask = 0.0, 0.0
        self.gen = torch.Generator(device=dev)
        self._handles = []
        self._install()

    # ------------------------------------------------------------------ hooks
    def _install(self):
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._head_hook(i)))
            self._handles.append(layer.mlp.register_forward_hook(self._mlp_hook(i)))
            self._handles.append(layer.register_forward_hook(self._skip_hook(i), with_kwargs=True))
        self._handles.append(self.model.model.embed_tokens.register_forward_hook(self._embed_hook))

    def _head_hook(self, i):
        def hook(module, args):
            m = self.head_mask[i]
            if m.requires_grad or not bool((m == 1).all()):
                x = args[0]
                shape = x.shape
                x = x.view(*shape[:-1], self.n_heads, self.head_dim) * m.to(x.dtype)[:, None]
                return (x.view(shape),) + tuple(args[1:])
        return hook

    def _mlp_hook(self, i):
        def hook(module, args, out):
            if i in self.mlp_off:
                return torch.zeros_like(out)
        return hook

    def _skip_hook(self, i):
        def hook(module, args, kwargs, out):
            if i in self.skip:
                hs = args[0] if args else kwargs["hidden_states"]
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
        return hook

    def _embed_hook(self, module, args, out):
        if self.embed_noise > 0:
            rms = out.float().pow(2).mean(-1, keepdim=True).sqrt()
            noise = torch.randn(out.shape, generator=self.gen, device=out.device) * rms * self.embed_noise
            out = (out.float() + noise).to(out.dtype)
        if self.embed_mask > 0:
            keep = torch.rand(out.shape[:2], generator=self.gen, device=out.device) >= self.embed_mask
            keep[:, 0] = True                                   # never mask BOS
            out = out * keep[..., None].to(out.dtype)
        return out

    # ------------------------------------------------------------- control
    def reset(self):
        self.head_mask = torch.ones_like(self.head_mask)
        self.mlp_off, self.skip = set(), set()
        self.embed_noise, self.embed_mask = 0.0, 0.0

    @contextlib.contextmanager
    def apply(self, heads=(), attn_layers=(), mlp_layers=(), skip_layers=(),
              embed_noise=0.0, embed_mask=0.0, seed=0):
        """Temporarily switch on a set of ablations."""
        self.reset()
        for l, h in heads:
            self.head_mask[l, h] = 0.0
        for l in attn_layers:
            self.head_mask[l, :] = 0.0
        self.mlp_off, self.skip = set(mlp_layers), set(skip_layers)
        self.embed_noise, self.embed_mask = embed_noise, embed_mask
        self.gen.manual_seed(seed)
        try:
            yield self
        finally:
            self.reset()

    def remove(self):
        for h in self._handles:
            h.remove()


def use_fp32_compute(model):
    """Keep weights stored in fp16 (fits a 4 GB GPU) but do all arithmetic in fp32.

    Some GPUs (e.g. GTX 16xx) run fp16 matrix multiplies several times slower than
    fp32, so each Linear upcasts its weight on the fly, and the embedding output is
    upcast so the residual stream, norms and attention all run in fp32.
    """
    import types
    import torch.nn.functional as F

    def linear_fp32(self, x):
        b = None if self.bias is None else self.bias.float()
        return F.linear(x.float(), self.weight.float(), b)

    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.forward = types.MethodType(linear_fp32, mod)
    model.model.embed_tokens.register_forward_hook(lambda m, a, out: out.float())
    return model
