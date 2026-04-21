# Model Architecture Summary

This document summarizes the full data flow, layer configuration, and LoRA support for the main policy network, combining the feature extraction base (`rl/networks/networkss.py`) and the final distribution wrapper (`rl/networks/model.py`).

## 1. Feature Extraction Base (`networkss.py`)

| Component         | Layer Name            | Layer Type         | Input Size | Input Name                 | Output Size | Output Name                | LoRA Support |
| :---------------- | :-------------------- | :----------------- | :--------- | :------------------------- | :---------- | :------------------------- | :----------- |
| Robot Embedding   | `robot_linear`        | `nn.Sequential`    | 9          | `robot_history` + `node`   | 256         | `robot_encoded`            | No           |
| HH Attn Embedding | `embedding_layer`     | `nn.Sequential`    | 12 / 17    | `human_states`             | 512         | `input_emb`                | No           |
| HH Projections    | `q, k, v_linear`      | `nn.Linear`        | 512        | `input_emb`                | 512         | `q, k, v`                  | No           |
| HH Interaction    | `multihead_attn`      | `MultiheadAttention`| 512        | `q, k, v`                  | 512         | `z`                        | No           |
| HH Output Squeeze | `hh_out_proj`         | `nn.Sequential`    | 512        | `hh_features` (z)          | 256         | `processed_human_features` | **Yes**      |
| HR Attn Query     | `robot_feature_proj`  | `nn.Linear`*       | 256        | `robot_encoded`            | 64          | `robot_embed`              | **Yes**      |
| HR Attn Key       | `human_feature_proj`  | `nn.Linear`*       | 256        | `processed_human_features` | 64          | `human_embed`              | **Yes**      |
| HR Attn Value     | (Identity/Pass)       | -                  | 256        | `processed_human_features` | 256         | `human_features_flat`      | -            |
| HR Context Logic  | `hr_attn`             | (Softmax/BMM)      | 64 / 256   | `robot_embed` / `v_input`  | 256         | `attended_human_features`  | -            |
| RNN Robot Enc     | `encoder_linear`      | `nn.Linear`        | 256        | `robot_encoded`            | 64          | `encoded_input`            | No           |
| RNN Social Enc    | `edge_attention_embed`| `nn.Linear`        | 256        | `attended_human_features`  | 64          | `h_edges_embedded`         | No           |
| Temporal Memory   | `gru` (in RNNBase)    | `nn.GRU`           | 128        | `concat_encoded`           | 128         | `x` (Hidden state)         | No           |
| RNN Projection    | `output_linear`       | `nn.Linear`        | 128        | `x`                        | 256         | `outputs`                  | No           |
| Actor Head        | `actor`               | `nn.Sequential`    | 256        | `x`                        | 256         | `hidden_actor`             | **Yes** (Both) |
| Critic Head       | `critic`              | `nn.Sequential`    | 256        | `x`                        | 256         | `hidden_critic`            | **Yes** (Both) |
| State Value       | `critic_linear`       | `nn.Linear`        | 256        | `hidden_critic`            | 1           | `value`                    | **Yes**      |

*\* `robot_feature_proj` and `human_feature_proj` are inside `ModuleList` in the `HRAttention` class.*

## 2. Policy Wrapper (`model.py`)

The final outputs of the base network (`hidden_actor`) are passed to a distribution wrapper. 

| Component         | Layer Name            | Layer Type         | Input Size | Input Name                 | Output Size | Output Name                | LoRA Support |
| :---------------- | :-------------------- | :----------------- | :--------- | :------------------------- | :---------- | :------------------------- | :----------- |
| Action Mean       | `dist.fc_mean`        | `nn.Linear`        | 256        | `hidden_actor`             | 2           | `action_mean`              | No**         |
| Action LogStd     | `dist.logstd`         | `AddBias`          | 2          | `action_mean`              | 2           | `action`                   | No           |

*\*\* If LoRA was ever applied to the final action output, it would be wrapped around `fc_mean` here.*

---

## Naming Migration Map (Backward Compatibility)

The following map is used in `train.py` to ensure compatibility with old checkpoints when loading weights into the new `networkss.py` architecture:

| Old Variable Name       | New Variable Name (in `networkss.py`) |
| :---------------------- | :------------------------------------ |
| `humanNodeRNN`          | `node_rnn`                            |
| `attn`                  | `hr_attn`                             |
| `spatial_attn`          | `hh_attn`                             |
| `spatial_linear`        | `hh_out_proj`                         |
| `temporal_edge_layer`   | `robot_feature_proj`                  |
| `spatial_edge_layer`    | `human_feature_proj`                  |
