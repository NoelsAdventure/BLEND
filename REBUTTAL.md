# Rebuttal Training Commands

For `--num-env-steps 10000000` with default `--num-processes 128` and `--num-steps 30`, the final checkpoint is `02603.pt`. Use `05207.pt` only for 20M-step backbone runs.

CUDA_VISIBLE_DEVICES=0 python train.py \
    --note Conservative_small \
    --robot-invisible \
    --load-path "" \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=1 python train.py \
    --note Conservative_medium \
    --robot-invisible \
    --load-path "" \
    --human_node_rnn_size 960 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 1920 \
    --human_node_embedding_size 480 \
    --human_human_edge_embedding_size 64 \
    --attention_size 480 \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=2 python train.py \
    --note Conservative_large \
    --robot-invisible \
    --load-path "" \
    --human_node_rnn_size 1984 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 7168 \
    --human_node_embedding_size 992 \
    --human_human_edge_embedding_size 64 \
    --attention_size 992 \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=3 python train.py \
    --note Fullfinetune_small \
    --robot-visible \
    --resume \
    --load-path trained_models/Ours_GST/checkpoints/05207.pt \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=x python train.py \
    --note LoRA_medium \
    --robot-visible \
    --use-lora \
    --lora-base-checkpoint trained_models/Conservative_medium/checkpoints/02603.pt \
    --human_node_rnn_size 960 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 1920 \
    --human_node_embedding_size 480 \
    --human_human_edge_embedding_size 64 \
    --attention_size 480 \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=x python train.py \
    --note Fullfinetune_medium \
    --robot-visible \
    --resume \
    --load-path trained_models/Conservative_medium/checkpoints/02603.pt \
    --human_node_rnn_size 960 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 1920 \
    --human_node_embedding_size 480 \
    --human_human_edge_embedding_size 64 \
    --attention_size 480 \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=x python train.py \
    --note LoRA_large \
    --robot-visible \
    --use-lora \
    --lora-base-checkpoint trained_models/Conservative_large/checkpoints/02603.pt \
    --human_node_rnn_size 1984 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 7168 \
    --human_node_embedding_size 992 \
    --human_human_edge_embedding_size 64 \
    --attention_size 992 \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=x python train.py \
    --note Fullfinetune_large \
    --robot-visible \
    --resume \
    --load-path trained_models/Conservative_large/checkpoints/02603.pt \
    --human_node_rnn_size 1984 \
    --human_human_edge_rnn_size 256 \
    --human_node_output_size 7168 \
    --human_node_embedding_size 992 \
    --human_human_edge_embedding_size 64 \
    --attention_size 992 \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=x python train.py \
    --note LoRA_small \
    --robot-visible \
    --use-lora \
    --lora-base-checkpoint trained_models/Conservative_small/checkpoints/02603.pt \
    --num-env-steps 10000000

CUDA_VISIBLE_DEVICES=x python train.py \
    --note Fullfinetune_small_v2 \
    --robot-visible \
    --resume \
    --load-path trained_models/Conservative_small/checkpoints/02603.pt \
    --num-env-steps 10000000
<!-- DONEEEE -->

