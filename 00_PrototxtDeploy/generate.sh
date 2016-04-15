#!/bin/bash
mkdir -p generated

cat ./header_normal.prototxt > generated/stage2.prototxt
cat ./toparam_header.prototxt >> generated/stage2.prototxt
sed 's/LABEL/100/g' ./toparam.prototxt >> generated/stage2.prototxt
sed 's/LABEL/010/g' ./toparam.prototxt >> generated/stage2.prototxt
sed 's/LABEL/001/g' ./toparam.prototxt >> generated/stage2.prototxt
cat ./param_to_img_header.prototxt >> generated/stage2.prototxt
sed 's/LABEL/100/g' ./param_to_img.prototxt >> generated/stage2.prototxt
sed 's/LABEL/010/g' ./param_to_img.prototxt >> generated/stage2.prototxt
sed 's/LABEL/001/g' ./param_to_img.prototxt >> generated/stage2.prototxt

cat ./header_normal.prototxt > generated/normal.prototxt
cat ./toparam_header.prototxt >> generated/normal.prototxt
sed 's/LABEL/100/g' ./toparam.prototxt >> generated/normal.prototxt
sed 's/LABEL/010/g' ./toparam.prototxt >> generated/normal.prototxt
sed 's/LABEL/001/g' ./toparam.prototxt >> generated/normal.prototxt
cat ./param_to_img_header.prototxt >> generated/normal.prototxt
sed 's/LABEL/100/g' ./param_to_img.prototxt >> generated/normal.prototxt
sed 's/LABEL/100/g' ./compare_img.prototxt >> generated/normal.prototxt
sed 's/LABEL/010/g' ./param_to_img.prototxt >> generated/normal.prototxt
sed 's/LABEL/010/g' ./compare_img.prototxt >> generated/normal.prototxt
sed 's/LABEL/001/g' ./param_to_img.prototxt >> generated/normal.prototxt
sed 's/LABEL/001/g' ./compare_img.prototxt >> generated/normal.prototxt
cat ./footer.prototxt >> generated/normal.prototxt

cat ./header_adversarial.prototxt > generated/adversarial.prototxt
cat ./toparam_header.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/100/g' ./toparam.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/010/g' ./toparam.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/001/g' ./toparam.prototxt >> generated/adversarial.prototxt
cat ./param_to_img_header.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/100/g' ./param_to_img.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/100/g' ./compare_img.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/010/g' ./param_to_img.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/010/g' ./compare_img.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/001/g' ./param_to_img.prototxt >> generated/adversarial.prototxt
sed 's/LABEL/001/g' ./compare_img.prototxt >> generated/adversarial.prototxt
cat ./footer.prototxt >> generated/adversarial.prototxt
cat ./loss.prototxt >> generated/adversarial.prototxt
