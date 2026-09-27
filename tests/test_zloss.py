"""z-loss: off by default and byte-identical, positive and formula-exact when on, and it pulls log Z toward 0."""

import torch
import torch.nn.functional as F

from slm.model.loss import IGNORE_INDEX, chunked_cross_entropy, logz_stats


def _case():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 16)
    w = torch.randn(50, 16)
    t = torch.randint(0, 50, (2, 5))
    t[0, 0] = IGNORE_INDEX
    return h, w, t


def test_zloss_off_is_the_plain_cross_entropy_sum():
    h, w, t = _case()
    loss, n = chunked_cross_entropy(h, w, t)
    logits = F.linear(h, w).reshape(-1, 50)
    ref = F.cross_entropy(logits, t.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")
    assert torch.allclose(loss, ref) and n.item() == 9
    assert torch.allclose(chunked_cross_entropy(h, w, t, chunk_size=3)[0], ref)


def test_zloss_on_adds_the_squared_log_normaliser_per_valid_token():
    h, w, t = _case()
    base, _ = chunked_cross_entropy(h, w, t)
    with_z, n = chunked_cross_entropy(h, w, t, z_loss=1e-2)
    logz = torch.logsumexp(F.linear(h, w).reshape(-1, 50), dim=-1)
    valid = t.reshape(-1) != IGNORE_INDEX
    expected = base + 1e-2 * (logz.square() * valid).sum()
    assert torch.allclose(with_z, expected) and with_z > base
    assert torch.allclose(chunked_cross_entropy(h, w, t, chunk_size=4, z_loss=1e-2)[0], expected), "chunked path agrees"


def test_zloss_gradient_pushes_log_z_toward_zero():
    h, w, t = _case()
    w = w + 3.0  # a shared logit offset: cross-entropy cannot see it, z-loss can
    w.requires_grad_(True)
    loss, _ = chunked_cross_entropy(h, w, t, z_loss=1.0)
    loss.backward()
    before = logz_stats(h, w.detach(), t)[0]
    w2 = (w - 0.01 * w.grad).detach()
    after = logz_stats(h, w2, t)[0]
    assert abs(after) < abs(before), "one step against the z-loss gradient shrinks the mean log Z"
