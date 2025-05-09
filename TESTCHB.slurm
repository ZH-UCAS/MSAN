#!/bin/bash
#SBATCH -J TESTDOU
#SBATCH --partition=q04
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=g42
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1


python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=1  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=2  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=3  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=5  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=6  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=7  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=8  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=9  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=10 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=11 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=13 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=14 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=16 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=17 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=18 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=20 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=21 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=22 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
python -u /share/home/code/test_SwinLSTM_Double_LSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=23 --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60
