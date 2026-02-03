
export SSD_ROOT="/home/alpahos/satassd/gamma_data_root"
export HDD_DATA_ROOT="/home/alpahos/8to/gamma_data/data_root/data"

export DEDUP_ZST="$SSD_ROOT/dedup/stageA_latest/deduped.jsonl.zst"
export TOK_DIR="$SSD_ROOT/tokenizers/ultimate_unigram_44032_v1"
export TOK="$TOK_DIR/tokenizer.model"

export PACK_SMOKE="$SSD_ROOT/packed/stageA_smoke_4096_5kseq"
export PACK_DEBUG="$SSD_ROOT/packed/stageA_debug_4096_50kseq"
export PACK_TRAIN_SMALL="$SSD_ROOT/packed/stageA_train_small_4096_200kseq"

echo "[env] SSD_ROOT=$SSD_ROOT"
echo "[env] DEDUP_ZST=$DEDUP_ZST"
echo "[env] TOK=$TOK"
echo "[env] PACK_SMOKE=$PACK_SMOKE"

# Ensure HDD manifest resolution sees stageA_latest
mkdir -p "$HDD_DATA_ROOT/dedup"
rm -f "$HDD_DATA_ROOT/dedup/stageA_latest"
ln -s "$SSD_ROOT/dedup/stageA_latest" "$HDD_DATA_ROOT/dedup/stageA_latest"

ls -lah "$HDD_DATA_ROOT/dedup/stageA_latest" | sed -n '1,120p'
ls -lh "$DEDUP_ZST" "$TOK" | sed -n '1,120p'
