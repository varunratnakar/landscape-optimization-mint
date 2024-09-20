#!/bin/sh

if [ $# -lt 2 ]; then
    echo "need to specify port number and max"
    exit 1
fi

n=$1
max=$2

port=878$n

ulimit -n 10000

nohup dask scheduler --port $port 2>&1 >/dev/null &
spid=$!

nohup dask worker --nthreads 1 --nworkers 64 --memory-limit 3.8GB tcp://localhost:$port 2>&1 >/dev/null &
wpid=$!

find results/hazard_config/ -type f | sort | xargs ./preprocess_hazard_subtrunks.py -s tcp://localhost:$port -m $max -c

kill $wpid
kill $spid
