#!/usr/bin/env bash
# Run on the destination only. Never installs drivers or overwrites environments.
set -euo pipefail
mode=${1:-plan}
bundle=${2:-/data/habitat_migration}
if [[ "$bundle" != /* || "$bundle" == / ]]; then
  echo 'Supply an absolute, dedicated migration bundle directory.' >&2
  exit 2
fi
case "$mode" in plan|restore|check) ;; *) echo "Usage: bash $0 {plan|restore|check} /absolute/bundle" >&2; exit 2 ;; esac
if [[ "$mode" == plan ]]; then
  echo "Bundle: $bundle"
  echo '1. Copy the full bundle, verify files, and create the source path links in README section 5.'
  echo '2. Run restore: extracts four archives into NEW envs/<name> directories and runs conda-unpack.'
  echo '3. Run check: checks GPU tensor execution and Habitat imports; this is NOT a rendering/model test.'
  echo 'No changes made. No driver installation or automatic dependency upgrades.'
  exit 0
fi
if [[ "$mode" == restore ]]; then
  # Validate every target before making the first change; never merge environments.
  for name in habitat navila navila-eval navid; do
    archive="$bundle/packed_envs/$name.tar.gz"
    target="$bundle/envs/$name"
    [[ -f "$archive" ]] || { echo "Missing archive: $archive" >&2; exit 1; }
    [[ ! -e "$target" && ! -L "$target" ]] || { echo "Refusing existing target: $target" >&2; exit 1; }
  done
  for name in habitat navila navila-eval navid; do
    target="$bundle/envs/$name"
    mkdir -p "$target"
    tar -xzf "$bundle/packed_envs/$name.tar.gz" -C "$target"
    "$target/bin/python" "$target/bin/conda-unpack"
    echo "Restored: $name"
  done
  echo 'Environment extraction complete. Run check next, then the README rendering/episode smoke tests.'
  exit 0
fi
uname -m
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for name in habitat navila navila-eval navid; do
  echo "Checking: $name"
  "$bundle/envs/$name/bin/python" -c 'import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones(4, device="cuda"); print("GPU tensor sum:", x.sum().item()); torch.cuda.synchronize()'
done
for name in habitat navila-eval; do
  "$bundle/envs/$name/bin/python" -c 'import habitat, habitat_sim; print("Habitat:", getattr(habitat, "__version__", "unknown"), habitat.__file__); print("Simulator:", habitat_sim.__file__)'
done
echo 'Basic checks passed. Rendering, model inference and full experiment reproduction remain separate checks.'
