#!/bin/bash
cd ~/pami_pilots
while [ ! -f exdark_DONE.flag ]; do sleep 30; done
sleep 5
python3 agg_exdark.py > exdark_RESULTS.txt 2>&1
echo FINALIZE_EXDARK_DONE > exdark_finalize_DONE.flag
