import torch

from crs.compress.apply import compress_model, target_linears
from crs.compress.kvcache import FakeQuantKVCache, fake_quant_lastdim, fake_quant_tokens
from crs.compress.lowrank import asvd_truncate, svd_truncate
from crs.compress.prune import magnitude_prune, wanda_prune
from crs.compress.quant import fake_quant_rtn


def test_rtn_group_levels():
    w = torch.randn(16, 256)
    for bits in (3, 4):
        q = fake_quant_rtn(w, bits, 128)
        for g in q.reshape(16, 2, 128).reshape(-1, 128):
            assert g.unique().numel() <= 2**bits
        assert (q - w).abs().max() < (w.max() - w.min())  # sane error
    pc = fake_quant_rtn(w, 3, -1)
    assert all(row.unique().numel() <= 8 for row in pc)


def test_rtn_sym():
    q = fake_quant_rtn(torch.randn(4, 64), 4, 64, sym=True)
    assert all(r.unique().numel() <= 16 for r in q)


def test_pruning_exact_fraction():
    w = torch.randn(32, 100)
    m = magnitude_prune(w, 0.3)
    assert (m == 0).sum().item() == 960
    wa = wanda_prune(w, torch.rand(100) + 0.1, 0.5)
    assert all((row == 0).sum().item() == 50 for row in wa)


def test_svd_rank():
    w = torch.randn(40, 30)
    t = svd_truncate(w, 0.5)
    assert torch.linalg.matrix_rank(t).item() == 15
    assert torch.equal(svd_truncate(w, 1.0), w)


def test_kv_fake_quant_levels():
    x = torch.randn(1, 2, 5, 128)
    q = fake_quant_lastdim(x, 4, 64)
    assert all(g.unique().numel() <= 16 for g in q.reshape(-1, 64))
    assert torch.equal(fake_quant_lastdim(x, 16, 64), x)


def test_compress_model_targets_decoder_linears(tiny):
    names = target_linears(tiny)
    assert names and all(".layers." in n for n in names) and "lm_head" not in names
    head = tiny.lm_head.weight.clone()
    calib = [torch.randint(0, 1000, (1, 32)) for _ in range(2)]
    stats = compress_model(tiny, {"method": "wanda", "sparsity": 0.5}, calib)
    assert abs(stats["zero_fraction"] - 0.5) < 0.01
    assert torch.equal(tiny.lm_head.weight, head)


def test_kv_cache_generate(tiny, tok):
    ids = torch.tensor([tok.encode("Hello there")])
    out = tiny.generate(ids, max_new_tokens=5, do_sample=False, past_key_values=FakeQuantKVCache(2, 16, key_group=4, residual=2),
                        pad_token_id=tok.eos_token_id)
    assert out.shape[1] == ids.shape[1] + 5


def test_kv_per_channel_handles_outlier_channels():
    """Keys with one huge channel (as in Qwen): per-channel int4 keeps the other channels,
    per-token int4 flattens them. This is why int4 output was gibberish."""
    torch.manual_seed(0)
    k = torch.randn(1, 2, 64, 128)
    k[..., 7] *= 100
    rest = [c for c in range(128) if c != 7]
    err_tok = (fake_quant_lastdim(k, 4, 64) - k)[..., rest].abs().mean()
    err_ch = (fake_quant_tokens(k, 4) - k)[..., rest].abs().mean()
    assert err_ch < err_tok / 5


def test_kv_cache_residual_window():
    """Only whole blocks older than the newest `residual` tokens are quantized."""
    c = FakeQuantKVCache(bits=2, group_size=16, key_group=8, residual=10)
    k, v = torch.randn(1, 2, 30, 32), torch.randn(1, 2, 30, 32)
    ko, vo = c.update(k.clone(), v.clone(), 0)
    assert c.n_quant[0] == 16  # 30 - 10 = 20 old tokens -> 2 whole blocks of 8
    assert torch.equal(ko[:, :, 16:], k[:, :, 16:]) and torch.equal(vo[:, :, 16:], v[:, :, 16:])
    assert not torch.equal(ko[:, :, :16], k[:, :, :16])
    assert all(ko[0, 0, :8, ch].unique().numel() <= 4 for ch in range(32))  # 2 bits per channel block
    ko, _ = c.update(torch.randn(1, 2, 6, 32), torch.randn(1, 2, 6, 32), 0)  # 36 tokens -> 3 blocks
    assert c.n_quant[0] == 24 and ko.shape[-2] == 36


def test_asvd_rank_and_activation_weighting():
    torch.manual_seed(0)
    w = torch.randn(48, 64)
    act = torch.ones(64)
    act[:4] = 50.0  # a few input channels carry most of the activation
    a = asvd_truncate(w, act, 0.5)
    p = svd_truncate(w, 0.5)
    assert torch.linalg.matrix_rank(a) == 24  # half of min(48, 64)
    # ASVD protects the heavily used channels: smaller error where it matters
    x = torch.randn(256, 64) * act
    assert ((x @ (a - w).T).norm()) < ((x @ (p - w).T).norm())


def test_awq_groups_no_worse_than_rtn(tiny):
    """AWQ groups q/k/v and gate/up, and each group's output error on the calibration inputs is
    no worse than plain RTN's (alpha 0 and no clipping are among the options searched)."""
    from crs.compress.awq import awq_quantize, collect_inputs, layer_groups

    torch.manual_seed(0)
    linears = target_linears(tiny)
    groups = layer_groups(linears)
    shorts = sorted(tuple(n.rsplit(".", 1)[-1] for n in g) for g in groups if ".layers.0." in g[0])
    assert shorts == [("down_proj",), ("gate_proj", "up_proj"), ("o_proj",), ("q_proj", "k_proj", "v_proj")]
    calib = [torch.randint(0, 1000, (1, 64)) for _ in range(4)]
    new, stats = awq_quantize(tiny, linears, calib, bits=3, group_size=32, n_tokens=256)
    assert set(new) == set(linears) and 0 <= stats["awq_mean_alpha"] < 1 and 0.5 <= stats["awq_mean_clip"] <= 1
    feats = collect_inputs(tiny, linears, groups, calib, 256)
    for g in groups:  # the scale is shared by a group, so the guarantee is on the group's total error
        x = feats[g[0]][1].float()
        e_awq = e_rtn = 0.0
        for name in g:
            w = linears[name].weight.data.float()
            e_awq += float((x @ (new[name] - w).T).pow(2).sum())
            e_rtn += float((x @ (fake_quant_rtn(w, 3, 32) - w).T).pow(2).sum())
        assert e_awq <= e_rtn * 1.001, g


def test_awq_protects_large_channels():
    """With a few huge input channels, the scale search picks alpha > 0 and beats RTN, while
    still rounding to 2**bits levels per group."""
    from crs.compress.awq import _scales, search_scale

    torch.manual_seed(0)
    w = torch.randn(64, 128)
    x = torch.randn(512, 128)
    x[:, :4] *= 30
    absmean = x.abs().mean(0)
    s, a = search_scale(x, absmean, w, 3, 64)
    assert a > 0 and torch.allclose(s, _scales(absmean, a))
    wq = fake_quant_rtn(w * s, 3, 64) / s
    err = lambda q: (x @ (q - w).T).pow(2).mean()  # noqa: E731
    assert err(wq) < 0.9 * err(fake_quant_rtn(w, 3, 64))
    # in the scaled space each group of 64 still holds at most 2**3 values
    assert all(g.unique().numel() <= 8 for g in fake_quant_rtn(w * s, 3, 64).reshape(-1, 64))


def test_compress_model_awq(tiny):
    calib = [torch.randint(0, 1000, (1, 48)) for _ in range(2)]
    head = tiny.lm_head.weight.clone()
    stats = compress_model(tiny, {"method": "awq", "bits": 4, "group_size": 32, "n_tokens": 64}, calib)
    assert stats["n_linear"] == 14 and "awq_mean_alpha" in stats and stats["mean_rel_frob_error"] > 0
    assert torch.equal(tiny.lm_head.weight, head)
