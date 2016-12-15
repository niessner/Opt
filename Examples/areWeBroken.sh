#!/bin/bash

# Author: Michael Mara (mmara@cs.stanford.edu)
# Created 2015-11-20
# Last Edited 2016-12-11


foldernames=(ImageWarping IntrinsicLP MeshDeformationARAP MeshDeformationED MeshDeformationLARAP MeshSmoothingLaplacian MeshSmoothingLaplacianCOT OpticalFlow PoissonImageEdition RobustMeshDeformationARAP ShapeFromShadingSimple SmoothingLaplacianFloat4Graph SmoothingLP)
programnames=(imagewarping imagewarping meshdeformation meshdeformation meshsmoothing meshsmoothing opticalflow imageediting meshdeformation shapefromshading imagewarping imagewarping)

#foldernames=(MeshDeformationED)
#programnames=(meshdeformation)


for index in ${!foldernames[*]}
do
    cd ${foldernames[$index]}
    #make clean
    make
    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
    ./${programnames[$index]}
    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
    cd ..
done
