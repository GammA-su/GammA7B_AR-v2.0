cat <<'EOF' > env.local.sh
# === EDIT THESE 2 LINES ONLY ===
export GAMMA7B_DATA_ROOT="/home/alpahos/8to/data"     # HDD
export GAMMA7B_SSD_ROOT="/home/alpahos/satassd/data"    # SATA SSD (1TB)

# Keep Hugging Face caches OFF the NVMe
export HF_HOME="$GAMMA7B_DATA_ROOT/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"

# Avoid tmp surprises
export TMPDIR="$GAMMA7B_DATA_ROOT/tmp"
EOF

# load it for this shell
source ./env.local.sh

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
