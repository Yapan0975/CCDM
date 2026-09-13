#!/bin/bash
# run_pg_campaign.sh - Poisson-Gaussian re-run of the composite track (Tables 2/6, the lambda
# surface, the F2 architecture table and the OneRestore retrain), so that every composite result
# uses the noise model stated in the paper.
# Uses ONLY GPU 0 and 1, three jobs per card (CPU-side synthesis is the bottleneck, so three
# concurrent jobs on one card run at ~1.3x the single-job step time).
# Jobs are read from pg_queue.txt ("TAG|command"). The queue is re-read whenever a slot frees,
# so jobs can be appended while the campaign runs. A job is skipped when TAG.done, TAG.running
# or TAG.failed exists. The launcher exits once nothing is pending or running and
# pg_queue.closed exists.
cd ~/Documents/yping/composite-restore
SLOTS=(0 0 0 1 1 1)
declare -A PID TAGOF
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

next_job() {
  while IFS='|' read -r tag cmd; do
    [ -z "$tag" ] && continue
    case "$tag" in \#*) continue ;; esac
    if [ -f "$tag.done" ] || [ -f "$tag.running" ] || [ -f "$tag.failed" ]; then continue; fi
    echo "$tag|$cmd"; return 0
  done < pg_queue.txt
  return 1
}

while true; do
  running=0
  for i in "${!SLOTS[@]}"; do
    p=${PID[$i]}
    if [ -n "$p" ]; then
      if kill -0 "$p" 2>/dev/null; then running=$((running+1)); continue; fi
      wait "$p"; rc=$?
      t=${TAGOF[$i]}
      rm -f "$t.running"
      if [ "$rc" -eq 0 ]; then
        touch "$t.done"; log "done $t (GPU${SLOTS[$i]})"
      else
        touch "$t.failed"; log "FAILED $t rc=$rc (GPU${SLOTS[$i]})"
      fi
      PID[$i]=""
    fi
    job=$(next_job) || continue
    IFS='|' read -r tag cmd <<< "$job"
    touch "$tag.running"
    CUDA_VISIBLE_DEVICES=${SLOTS[$i]} nohup bash -c "$cmd" > "$tag.log" 2>&1 &
    PID[$i]=$!; TAGOF[$i]=$tag; running=$((running+1))
    log "launch $tag on GPU${SLOTS[$i]} (slot $i, pid ${PID[$i]})"
    sleep 8
  done
  if [ "$running" -eq 0 ] && ! next_job > /dev/null && [ -f pg_queue.closed ]; then
    log "campaign finished"; echo PG_CAMPAIGN_DONE > pg_campaign_DONE.flag; break
  fi
  sleep 30
done
