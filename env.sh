# Source this before working: `source env.sh`
#
# nanoserve uses a project-local venv (.venv) created with --system-site-packages
# so it inherits torch/triton from /opt/pytorch but adds `transformers` in
# isolation, without mutating that shared environment.
#
# NOTE: pip pulls the latest torch (currently 2.14.0+cu130) into the venv on each
# install. Fine for this engine (stock HF ops, no custom CUDA compilation); pin
# torch/transformers/accelerate if you need reproducibility.

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
