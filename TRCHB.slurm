#!/bin/bash
#SBATCH -J TRCHB
#SBATCH --partition=q04
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=g39
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1


#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=2  --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=3  --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=5  --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=6  --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=7  --target_preictal_interval=60 --step_preictal=5
python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=8  --target_preictal_interval=60 --step_preictal=5
python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=9  --target_preictal_interval=60 --step_preictal=5
python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=10 --target_preictal_interval=60 --step_preictal=5
python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=11 --target_preictal_interval=60 --step_preictal=5
python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=13 --target_preictal_interval=60 --step_preictal=5
python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=14 --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=16 --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=17 --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=18 --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=20 --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=21 --target_preictal_interval=60 --step_preictal=5
#python -u /share/home/code/train_SwinLSTM_DP_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=22 --target_preictal_interval=60 --step_preictal=5
