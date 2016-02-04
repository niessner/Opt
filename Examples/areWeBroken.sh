#!/bin/bash

# Author: Michael Mara (mmara@cs.stanford.edu)
# Created 11-20-2015
# Last Edited 02-04-2016


foldernames=(SmoothingLaplacianFloat4Graph ImageWarping ShapeFromShadingSimple MeshDeformationARAP MeshDeformationED OpticalFlow PoissonImageEditing MeshSmoothingLaplacianCOT)
programnames=(meshsmoothing imagewarping shapefromshading meshdeformation meshdeformation opticalflow imageediting meshsmoothing)

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
