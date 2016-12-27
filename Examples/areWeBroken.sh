#!/bin/bash

# Author: Michael Mara (mmara@cs.stanford.edu)
# Created 2015-11-20
# Last Edited 2016-12-18

#TODO: Add smoothinglp
foldernames=(ImageWarping IntrinsicLP MeshDeformationARAP MeshDeformationED MeshDeformationLARAP MeshSmoothingLaplacian MeshSmoothingLaplacianCOT OpticalFlow PoissonImageEditing RobustMeshDeformationARAP ShapeFromShadingSimple SmoothingLaplacianFloat4Graph)
programnames=(imagewarping intrinsiclp meshdeformation meshdeformation meshdeformation meshsmoothing opticalflow imageediting meshdeformation shapefromshading imagewarping)

#foldernames=(MeshDeformationED)
#programnames=(meshdeformation)


for index in ${!foldernames[*]}
do
    cd ${foldernames[$index]}
    #make clean
    echo ${foldernames[$index]}
    make
    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
    cd ..
done

echo "==--All compile!--=="

for index in ${!foldernames[*]}
do
    cd ${foldernames[$index]}
    #make clean
    echo ${foldernames[$index]}
    ./${programnames[$index]}
    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
    cd ..
done

echo "==--All run!--=="
