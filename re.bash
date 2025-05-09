files=($(ls -1 *_patient4_10.pth))
for ((i=0; i<${#files[@]}; i++)); do
  new_name=$(echo ${files[$i]} | sed -E "s/_patient4_10.pth/_patient4_${i}_step_preictal10.pth/")
  mv "${files[$i]}" "$new_name"
done



cd /share/home/code//model/XW90/MONSTBX/3/stft/
files=($(ls -t *_patient3_*.pth | head -n 22))
for ((i=0; i<${#files[@]}; i++)); do
  new_name=$(echo ${files[$i]} | sed -E "s/_patient3_([0-9]+)_step_preictal([0-9]+).pth/_patient4_$i_step_preictal\2.pth/")
  mv "${files[$i]}" "/share/home/code/model/XW90/MONSTBX/4/stft/$new_name"
done
