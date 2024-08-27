import timesfm
import torch
import os

context_len = 1024
horizon_len = 1
model = timesfm.TimesFm(
context_len=context_len, # changable
horizon_len=horizon_len, # changable
input_patch_len=32,
output_patch_len=128,
num_layers=20,
model_dims=1280,
backend="gpu",
)
model.load_from_checkpoint(checkpoint_path="/YOUR_PATH/timesfm-1.0-200m/checkpoints")
jax_ckpt = model._train_state

if not os.path.exists('../ckpt'):
    os.makedirs('../ckpt')

freq_emb_ckpt = {}
freq_emb_ckpt['emb_var.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['freq_emb']['emb_var'])
torch.save(freq_emb_ckpt, '../ckpt/freq_emb.pt')

horizon_ff_layer_ckpt = {}
horizon_ff_layer_ckpt['hidden_layer.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['horizon_ff_layer']['hidden_layer']['linear']['w']).permute(1, 0)
horizon_ff_layer_ckpt['hidden_layer.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['horizon_ff_layer']['hidden_layer']['bias']['b'])
horizon_ff_layer_ckpt['output_layer.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['horizon_ff_layer']['output_layer']['linear']['w']).permute(1, 0)
horizon_ff_layer_ckpt['output_layer.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['horizon_ff_layer']['output_layer']['bias']['b'])
horizon_ff_layer_ckpt['residual_layer.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['horizon_ff_layer']['residual_layer']['linear']['w']).permute(1, 0)
horizon_ff_layer_ckpt['residual_layer.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['horizon_ff_layer']['residual_layer']['bias']['b'])
torch.save(horizon_ff_layer_ckpt, '../ckpt/horizon_ff_layer.pt')

input_ff_layer_ckpt = {}
input_ff_layer_ckpt['hidden_layer.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['input_ff_layer']['hidden_layer']['linear']['w']).permute(1, 0)
input_ff_layer_ckpt['hidden_layer.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['input_ff_layer']['hidden_layer']['bias']['b'])
input_ff_layer_ckpt['output_layer.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['input_ff_layer']['output_layer']['linear']['w']).permute(1, 0)
input_ff_layer_ckpt['output_layer.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['input_ff_layer']['output_layer']['bias']['b'])
input_ff_layer_ckpt['residual_layer.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['input_ff_layer']['residual_layer']['linear']['w']).permute(1, 0)
input_ff_layer_ckpt['residual_layer.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['input_ff_layer']['residual_layer']['bias']['b'])
torch.save(input_ff_layer_ckpt, '../ckpt/input_ff_layer.pt')

stack_transformer_ckpt = {}
for i in range(20):
    stack_transformer_ckpt[f'layers.{i}.attention.scale_query.per_dim_scale'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['per_dim_scale']['per_dim_scale'])
    stack_transformer_ckpt[f'layers.{i}.attention.query.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['query']['w']).reshape(1280, 1280).permute(1, 0)
    stack_transformer_ckpt[f'layers.{i}.attention.query.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['query']['b']).reshape(-1)
    stack_transformer_ckpt[f'layers.{i}.attention.key.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['key']['w']).reshape(1280, 1280).permute(1, 0)
    stack_transformer_ckpt[f'layers.{i}.attention.key.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['key']['b']).reshape(-1)
    stack_transformer_ckpt[f'layers.{i}.attention.value.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['value']['w']).reshape(1280, 1280).permute(1, 0)
    stack_transformer_ckpt[f'layers.{i}.attention.value.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['value']['b']).reshape(-1)
    stack_transformer_ckpt[f'layers.{i}.attention.post.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['post']['w']).reshape(1280, 1280)
    stack_transformer_ckpt[f'layers.{i}.attention.post.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['self_attention']['post']['b']).reshape(-1)
    stack_transformer_ckpt[f'layers.{i}.feed_forward.layer_norm.scale'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['ff_layer']['layer_norm']['scale'])
    stack_transformer_ckpt[f'layers.{i}.feed_forward.layer_norm.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['ff_layer']['layer_norm']['bias'])
    stack_transformer_ckpt[f'layers.{i}.feed_forward.linear1.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['ff_layer']['ffn_layer1']['linear']['w']).permute(1, 0)
    stack_transformer_ckpt[f'layers.{i}.feed_forward.linear1.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['ff_layer']['ffn_layer1']['bias']['b'])
    stack_transformer_ckpt[f'layers.{i}.feed_forward.linear2.weight'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['ff_layer']['ffn_layer2']['linear']['w']).permute(1, 0)
    stack_transformer_ckpt[f'layers.{i}.feed_forward.linear2.bias'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['ff_layer']['ffn_layer2']['bias']['b'])
    stack_transformer_ckpt[f'layers.{i}.layer_norm.scale'] = torch.tensor(jax_ckpt.mdl_vars['params']['stacked_transformer_layer'][f'x_layers_{i}']['layer_norm']['scale'])
torch.save(stack_transformer_ckpt, '../ckpt/stack_transformer.pt')
