#CHB-MIT

#preprocess one patient
preprocess:
	python EEG_preprocess/chb_preprocess.py --patient_id=1

#preprocess all patients
preprocess_chb_stft15_30:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python EEG_preprocess/chb_preprocess.py --patient_id=$$id --doing_STFT=True --window_length=30 --preictal_step=5 --interictal_step=30 --doing_lowpass_filter=False --target_preictal_interval=15 > Preprocess_stft_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

train_chb30_lstm_BX:
	for id in  1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -u train_SwinLSTM.py --dataset_name=CHB30 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=$$id --target_preictal_interval=30> TrainMONSTBX_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

DP_train_chb30_lstm_BX:
	for id in  1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23 ; \
	do \
		python -u train_SwinLSTM_DP.py --dataset_name=CHB30 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=$$id --target_preictal_interval=30> TrainMONSTBX_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

DDP_train_chb30_lstm_BX:
	for id in  1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -m torch.distributed.launch --nproc_per_node=2 --use_env train_SwinLSTM_DDP.py --dataset_name=CHB30 --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=18 --patient_id=$$id --target_preictal_interval=30> TrainMONSTBX_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

DP_train_xw_lstm30BX:
	for id in 1 2 3  ; \
	do \
		python -u train_SwinLSTM_DP.py --dataset_name=XUANWU --model_name=MONSTBX --loss=CE --device_number=0 --ch_num=108 --patient_id=$$id  --target_preictal_interval=30> TrainMONSTBX_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

#eval on one patient
eval:
	python test.py --dataset_name=CHB --model_name=TA_STS_ConvNet --device_number=1 --patient_id=1 --ch_num=18 --moving_average_length=9

#eval on all patients
eval_chb:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python test.py --dataset_name=CHB --model_name=TA_STS_ConvNet30 --device_number=2 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=5> Test_$${id}_$(shell date '+%Y-%m-%d_%H:%M').txt; \
	done

eval_chb15_stft_X:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -u test_SwinTrans.py --dataset_name=CHB --model_name=STANX --device_number=4 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=30 --persistence_second=30> STANXTest_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

eval_chb15_stft_C:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -u test_SwinTrans.py --dataset_name=CHB --model_name=STANC --device_number=5 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=30 --persistence_second=30> STANCTest_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

eval_chb30_stft_X:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -u test_SwinTrans.py --dataset_name=CHB30 --model_name=STANX --device_number=6 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=30 --persistence_second=30 --target_preictal_interval=30> STANXTest30_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

eval_chb30_stft_C:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -u test_SwinTrans.py --dataset_name=CHB30 --model_name=STANC --device_number=7 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=30 --persistence_second=30 --target_preictal_interval=30> STANCTest30_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done


eval_chb30_lstm_CL:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -u test_SwinLSTM.py --dataset_name=CHB30 --model_name=MONSTL --device_number=3 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=30 --persistence_second=30 --target_preictal_interval=30> MONSTCLTest30_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

eval_chb30_lstm_BX:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -u test_SwinLSTM.py --dataset_name=CHB30 --model_name=MONSTBX --device_number=0 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=30 --persistence_second=30 --target_preictal_interval=30> MONSTBXTest30_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

eval_xw_lstm30BXP:
	for id in 1 2 3  ; \
	do \
		python -u test_SwinLSTM.py --dataset_name=XUANWU --model_name=MONSTBX --device_number=0 --patient_id=$$id --ch_num=0 --moving_average_length=9 --step_preictal=30 --persistence_second=30 --target_preictal_interval=30> MONSTBXTest30_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

eval_chb_lstm60BX_69:
	for id in 11 14 16; \
	do \
		python -u test_SwinLSTM_Double.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=6 --persistence_second=30 --target_preictal_interval=60> MONSTBXTest30_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done

eval_chb_lstm60BX_610:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python -u test_SwinLSTM.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=$$id --ch_num=18 --moving_average_length=10 --step_preictal=30 --persistence_second=30 --target_preictal_interval=60> MONSTBXTest30_$${id}_$(shell date '+%Y-%m-%d_%H:%M:%S').txt; \
	done


eval_chb30:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python test.py --dataset_name=CHB --model_name=TA_STS_ConvNet30 --device_number=4 --patient_id=$$id --ch_num=18 --moving_average_length=9 --step_preictal=5 > Test_$${id}_$(shell date '+%Y-%m-%d_%H:%M').txt; \
	done


evaluation:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23 ; \
	do \
	    python -u test_SwinLSTM_Double_all_T.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=$$id  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60 > Test_$${id}_$(shell date '+%Y-%m-%d_%H:%M').txt; \
	done

trcnn:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23 ; \
	do \
	    python -u test_SwinLSTM_Double.py --dataset_name=CHB60 --model_name=MONSTBX --device_number=0 --patient_id=$$id  --ch_num=18 --moving_average_length=9 --step_preictal=5 --persistence_second=5 --target_preictal_interval=60 > Test_$${id}_$(shell date '+%Y-%m-%d_%H:%M').txt; \
	done

