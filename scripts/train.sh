export CUDA_VISIBLE_DEVICES=2,3;
export BS=2;
USE_TF=0
python src/run_translation.py \
  --model_name_or_path t5-3b \
  --output_dir /local/nlpswordfish/tariq/fallacy/t53b_cont_lr4_prop_logic \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --do_train \
  --do_eval \
  --train_file data/single_dataset/logic_train+propaganda_train10.json \
  --validation_file data/single_dataset/logic_test.json \
  --learning_rate 1e-4 \
  --overwrite_output_dir \
  --max_source_length 1024 \
  --max_target_length 64 \
  --num_train_epochs 8 \
  --gradient_accumulation_steps 512 \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size $BS \
  --source_lang input \
  --target_lang target