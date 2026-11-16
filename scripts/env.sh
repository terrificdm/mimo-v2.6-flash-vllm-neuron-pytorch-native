# SPDX-License-Identifier: Apache-2.0
# Paths and environment shared by the scripts in this directory. Sourced, not executed.
# Override any variable by exporting it before you run a script, for example
#   export MIMO_ROOT=/mnt/nvme/mimo
# Everything large (venv, source checkout, weights, compile cache, logs) goes under MIMO_ROOT, which must be on a
# local disk (the instance-store NVMe RAID): the compile cache uses flock, which is not reliable on NFS.
# No secrets here: the weights are public and need no Hugging Face token.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO_DIR

export MIMO_ROOT=${MIMO_ROOT:-/mnt/nvme/mimo}
export VENV=${VENV:-$MIMO_ROOT/venv}
export SRC_DIR=${SRC_DIR:-$MIMO_ROOT/src}
export VLLM_NEURON_DIR=${VLLM_NEURON_DIR:-$SRC_DIR/vllm-neuron}

# Weights: one pinned revision of the original Hugging Face checkpoint (no conversion step).
export HF_HOME=${HF_HOME:-$MIMO_ROOT/hf}
export HF_REPO=XiaomiMiMo/MiMo-V2.6-Flash-MOPD
export HF_REVISION=b12f5573350bb1704513fcf119fca48e498b0423
export HF_SNAPSHOT=$HF_HOME/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD/snapshots/$HF_REVISION

# Compile caches. serve.sh sets VLLM_CACHE_ROOT=$CACHE_BASE/<code tree>/<profile>-<spec> for each start.
export CACHE_BASE=${CACHE_BASE:-$MIMO_ROOT/cache}
export NEURON_COMPILE_CACHE_URL=${NEURON_COMPILE_CACHE_URL:-$MIMO_ROOT/neuron-compile-cache}
export NKI_COMPILE_CACHE_URL=${NKI_COMPILE_CACHE_URL:-$MIMO_ROOT/neuron-compile-cache}

# Temporary files and the libtorch-neuronx debug directory on the NVMe (the default /tmp can fill the root disk).
export TMPDIR=${TMPDIR:-$MIMO_ROOT/tmp}
export TORCH_NEURONX_DEBUG_DIR=${TORCH_NEURONX_DEBUG_DIR:-$MIMO_ROOT/tmp/neuron_backend}

# Media the server may read through file:// URLs (--allowed-local-media-path on the 1M profiles).
export MEDIA_DIR=${MEDIA_DIR:-$REPO_DIR/assets}

export LOG_DIR=${LOG_DIR:-$MIMO_ROOT/logs}
export RUN_DIR=${RUN_DIR:-$MIMO_ROOT/run}

# The pinned Neuron DLAMI venv. Only read (its pip version and package list); never modified.
export DLAMI_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0

# These debug variables change the NEFF cache key and are not used by this deployment.
unset XLA_IR_DEBUG XLA_HLO_DEBUG NEURON_FRAMEWORK_DEBUG

export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH

die() { local rc=$1; shift; echo "ERROR: $*" >&2; exit "$rc"; }
log() { echo "[$(basename "$0")] $(date -u +%FT%TZ) $*"; }
mkdir -p "$TMPDIR" "$TORCH_NEURONX_DEBUG_DIR" "$LOG_DIR" "$RUN_DIR" 2>/dev/null || true
