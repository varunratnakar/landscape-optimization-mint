#!/bin/sh

# run multiple processing in parallel

# run 4 for chunks with < 2000 rasters

nohup ./run.sh 0 2000 &
j1=$!

nohup ./run.sh 1 2000 &
j2=$!

nohup ./run.sh 2 2000 &
j3=$!

nohup ./run.sh 3 2000 &
j4=$!

wait $j1
wait $j2
wait $j3
wait $j4


# run 2 for the rest

nohup ./run.sh 0 6000 &
j1=$!

nohup ./run.sh 1 6000 &
j2=$!

wait $j1
wait $j2
