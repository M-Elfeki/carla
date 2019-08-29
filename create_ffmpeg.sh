#!/bin/bash
OUTPUTPATH=$1
CAMERA=$2
OUT=$3


mkdir tmp_$OUT
cp ./$OUTPUTPATH/RGB/camera-$CAMERA-* tmp_$OUT
cp ./$OUTPUTPATH/LogarithmicDepth/camera-$CAMERA-* tmp_$OUT
cp ./$OUTPUTPATH/SemanticSegmentation/camera-$CAMERA-* tmp_$OUT
cp ./$OUTPUTPATH/InstanceSegmentation/camera-$CAMERA-* tmp_$OUT

ffmpeg -pattern_type glob -i 'tmp_myout.avi/camera-*.png' -filter_complex tile=2x2:margin=10:padding=4 -vcodec mpeg4 -b:v 1200k myout.avi
