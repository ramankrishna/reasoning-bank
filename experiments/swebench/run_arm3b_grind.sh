#!/usr/bin/env bash
# arm3b grind wrapper: per-seed fresh bank, loops sessions until each seed
# has 30 done cells, prunes docker between sessions if disk free < 25GB.
#
# Usage:
#   ANTHROPIC_API_KEY=... bash experiments/swebench/run_arm3b_grind.sh \
#       [seed0,seed1,seed2] [session_min] [free_gb_threshold]
#
# Defaults: seeds=0,1,2  session_min=180  free_gb_threshold=25
set -u

SEEDS_CSV="${1:-0,1,2}"
SESSION_MIN="${2:-180}"
FREE_GB_MIN="${3:-25}"

LOG=experiments/swebench/run_data/arm3b_grind.log
CELLS_DIR=experiments/swebench/run_data/phase3_cells

mkdir -p "$(dirname "$LOG")"
echo "===== arm3b grind start $(date -u +%FT%TZ) seeds=$SEEDS_CSV session_min=$SESSION_MIN free_gb_min=$FREE_GB_MIN =====" >>"$LOG"

prune_if_low_disk() {
    free_gb=$(df -g /System/Volumes/Data | awk 'NR==2 {print $4}')
    echo "[wrapper] disk free: ${free_gb}GB (min ${FREE_GB_MIN}GB)" >>"$LOG"
    if [ "${free_gb:-0}" -lt "$FREE_GB_MIN" ]; then
        echo "[wrapper] pruning docker (free=${free_gb}GB < ${FREE_GB_MIN}GB)" >>"$LOG"
        docker system prune -af --volumes >>"$LOG" 2>&1
        free_gb=$(df -g /System/Volumes/Data | awk 'NR==2 {print $4}')
        echo "[wrapper] disk free after prune: ${free_gb}GB" >>"$LOG"
    fi
}

count_done() {
    local seed="$1"
    ls "$CELLS_DIR" 2>/dev/null | grep -c "^arm3b_bank_persist__.*__s${seed}\.json$"
}

IFS=',' read -ra SEEDS <<<"$SEEDS_CSV"
for SEED in "${SEEDS[@]}"; do
    DB_PATH="experiments/swebench/run_data/phase3_arm3b_bank_s${SEED}.sqlite"
    echo "" >>"$LOG"
    echo "===== seed=${SEED} bank=${DB_PATH} =====" >>"$LOG"

    SESSION_NUM=0
    while :; do
        DONE=$(count_done "$SEED")
        echo "[wrapper] seed=${SEED} done=${DONE}/30 session=${SESSION_NUM}" >>"$LOG"
        if [ "$DONE" -ge 30 ]; then
            echo "[wrapper] seed=${SEED} COMPLETE" >>"$LOG"
            break
        fi

        prune_if_low_disk

        SESSION_NUM=$((SESSION_NUM+1))
        SESSION_LOG="experiments/swebench/run_data/arm3b_s${SEED}_session${SESSION_NUM}.log"
        echo "[wrapper] starting seed=${SEED} session=${SESSION_NUM} -> ${SESSION_LOG}" >>"$LOG"

        SESSION_MAX_MINUTES="$SESSION_MIN" \
        PHASE3_ARMS=arm3b_bank_persist \
        PHASE3_SEEDS="$SEED" \
        PHASE3_BANK_DB_PATH="$DB_PATH" \
            uv run python experiments/swebench/run_phase3_resumable.py >"$SESSION_LOG" 2>&1
        SESSION_EXIT=$?
        echo "[wrapper] seed=${SEED} session=${SESSION_NUM} exit=${SESSION_EXIT}" >>"$LOG"

        # Defensive: if the session ended without making *any* progress,
        # back off briefly before retrying so we don't tight-loop on a hard error.
        NEW_DONE=$(count_done "$SEED")
        if [ "$NEW_DONE" -eq "$DONE" ]; then
            echo "[wrapper] WARN seed=${SEED} session=${SESSION_NUM} made no progress; backing off 60s" >>"$LOG"
            sleep 60
            # If second consecutive session makes no progress, abort this seed.
            if [ "$SESSION_NUM" -ge 2 ]; then
                LAST_LOG_TAIL=$(tail -20 "$SESSION_LOG" 2>/dev/null)
                echo "[wrapper] ABORT seed=${SEED}; last session log tail:" >>"$LOG"
                echo "$LAST_LOG_TAIL" >>"$LOG"
                break
            fi
        fi
    done
done

echo "" >>"$LOG"
echo "===== arm3b grind end $(date -u +%FT%TZ) =====" >>"$LOG"
for SEED in "${SEEDS[@]}"; do
    DONE=$(count_done "$SEED")
    echo "  seed=${SEED}: ${DONE}/30" >>"$LOG"
done
