#!/usr/bin/env bash
# Run run_pipeline.sh detached (survives closing the session). Log: $EXP/pipeline.log
EXP="${EXP:-/home/mtrack/exp_v2items}"
mkdir -p "$EXP"
setsid nohup bash "$(dirname "$0")/run_pipeline.sh" >> "$EXP/pipeline.log" 2>&1 < /dev/null &
echo $! > "$EXP/pipeline.pid"
echo "Started (PID $!). Follow: tail -f $EXP/pipeline.log   Stop: kill -- -\$(cat $EXP/pipeline.pid)"
