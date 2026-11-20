import torch

from crs.compress.apply import compress_model, target_linears
from crs.compress.kvcache import FakeQuantKVCache, fake_quant_lastdim
from crs.compress.lowrank import svd_truncate
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
    out = tiny.generate(ids, max_new_tokens=5, do_sample=False, past_key_values=FakeQuantKVCache(2, 16),
                        pad_token_id=tok.eos_token_id)
    assert out.shape[1] == ids.shape[1] + 5
