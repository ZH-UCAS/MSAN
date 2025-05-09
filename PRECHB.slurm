#!/bin/bash
#SBATCH -J MONSTERKAGGLE
#SBATCH --partition=q04
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=g40
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1

python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=1  --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=2  --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=3  --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=5  --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=6  --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=7  --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=8  --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=9  --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=10 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=11 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=13 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=14 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=16 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=17 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=18 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=20 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=21 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=22 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
python -u /share/home/code/EEG_preprocess/chb_preprocess_noplt.py --patient_id=23 --doing_STFT=True --doing_noise=True --window_length=30 --preictal_step=5 --interictal_step=30 --target_preictal_interval=60 --data_path=CHB60
