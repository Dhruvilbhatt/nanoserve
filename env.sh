# Source this before working: `source env.sh`
#
# nanoserve uses a project-local venv (.venv) created with --system-site-packages
# so it inherits torch/triton from /opt/pytorch but adds `transformers` in
# isolation, without mutating that shared environment.
#
# NOTE: pip pulls a newer torch (currently 2.14.0+cu130; it grabs the latest on
# each install) into the venv than the gpu-kernels repo uses (/opt/pytorch's
# 2.9.1+cu130). Fine for this engine (stock HF ops, no custom CUDA compilation);
# pin torch/transformers/accelerate if you need reproducibility. Only needs
# reconciling at experiment 4, when the gpu-kernels JIT kernels get wired in.

_MS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ ! -d "$_MS_ROOT/.venv" ]; then
  echo "no .venv yet — create it with:"
  echo "  /opt/pytorch/bin/python3 -m venv --system-site-packages $_MS_ROOT/.venv"
  echo "  $_MS_ROOT/.venv/bin/pip install transformers accelerate"
  return 1 2>/dev/null || exit 1
fi

source "$_MS_ROOT/.venv/bin/activate"
export PYTHONPATH="$_MS_ROOT:$PYTHONPATH"   # so `import engine` works from eval/ and bench/

echo "nanoserve env ready:"
echo "  python : $(command -v python)"
python - <<'PY'
import torch, transformers
print(f"  torch  : {torch.__version__} (cuda={torch.cuda.is_available()})")
print(f"  transformers: {transformers.__version__}")
PY
