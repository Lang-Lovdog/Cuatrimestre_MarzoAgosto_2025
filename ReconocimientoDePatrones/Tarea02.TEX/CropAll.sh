#!/bin/bash

CropBin="CPP/build/bin/matcrop"

for carpeta in ../imagenes/*; do
  if [ -d "$carpeta" ]; then
    i=0
    j=0
    while [ $i -le 10 ]; do
      falla="$carpeta/Falla_1_${i}_${j}.tif"
      mask="$carpeta/Mask_${i}_${j}.tif"
      if [ $j -gt 10 ]; then i=$((i+1)); j=0; else j=$((j+1)); fi
      if [ -f "$falla" ] && [ -f "$mask" ]; then
        $CropBin "$falla" "$mask"
      fi
    done
  fi
done

mv ../imagenes/clase1_sano/*.png ../imagenes/Cropped/Clase1-Sano/
mv ../imagenes/Clase2-Err1/*.png ../imagenes/Cropped/Clase2-Err1/
mv ../imagenes/Clase3Err2/*.png  ../imagenes/Cropped/Clase3-Err2/
mv ../imagenes/Clase4-Err3/*.png ../imagenes/Cropped/Clase4-Err3/
