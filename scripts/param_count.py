"""Print exact parameter counts and the training memory budget for model configs.

    python scripts/param_count.py configs/model/*.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm.config import ModelConfig, load_config  # noqa: E402
from slm.model import Transformer  # noqa: E402


def main(paths):
    print(f"{'config':22s} {'total':>13s} {'non-embed':>13s} {'embed':>12s} {'fp32 weights+grads+Adam':>26s}")
    for p in paths:
        cfg = load_config(ModelConfig, p)
        m = Transformer(cfg)
        total = m.num_params()
        ne = m.num_params(non_embedding=True)
        # fp32 master weights (4) + fp32 grads (4) + Adam m,v (8) = 16 bytes/param
        print(f"{Path(p).stem:22s} {total:13,d} {ne:13,d} {total - ne:12,d} {total * 16 / 2**30:22.2f} GiB")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(str(p) for p in Path("configs/model").glob("*.yaml")))
