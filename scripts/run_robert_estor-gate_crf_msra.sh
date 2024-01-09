CURRENT_DIR=`pwd`
export CUDA_VISIBLE_DEVICES=2
BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
ROBERTA_LARGE_DIR=$CURRENT_DIR/prev_trained_model/roberta-large-chinese
TRAINED_DIR=$CURRENT_DIR/outputs/msra_output/robert_pos_add_crf/checkpoint-3340
DATA_DIR=$CURRENT_DIR/datasets
OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="msra"
#
python main.py \
  --backbone=bert_estor_crf \
  --model_name_or_path=$ROBERTA_LARGE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --pos_learning_rate=5e-5\
  --crf_learning_rate=1e-3 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42 \
  --local_rank=-1 \
  --num_train_epochs=20 \
  --freeze_backbone_epoch=10\
  --tagging_rate=1\
  --enumerate_mode=gate\
  --gate_scaling_rate=0.8\
  --gate_dropout_rate=0.5\
  --if_add_a_self_attention=1\
  --if_merge_by_add=1\
  --if_contrastive_learn=1\
  --contrastive_alpha=0.1\

