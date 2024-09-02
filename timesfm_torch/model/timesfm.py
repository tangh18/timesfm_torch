import torch
from timesfm_torch.model.model import *
from timesfm_torch.model.utils import *

class TimesFm(nn.Module):
    def __init__(self, context_len, horizon_len=128, input_patch_len=32, output_patch_len=128, num_layers=20, model_dims=1280, num_outputs=10):
        super().__init__()
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.num_outputs = num_outputs

        self.freq_emb = Embedding(num_classes=3, input_dims=model_dims)
        self.position_emb = PositionalEmbedding(embedding_dims=model_dims).to('cuda')
        self.horizon_ff_layer = ResidualBlock(input_dims=model_dims, hidden_dims=model_dims, output_dims=model_dims).to('cuda')
        self.input_ff_layer = ResidualBlock(input_dims=64, hidden_dims=model_dims, output_dims=model_dims).to('cuda')
        self.stacked_transformer_layer = Transformer(num_layers=20, d_model=model_dims, num_heads=16, hidden_dim=model_dims).to('cuda')

    def load_from_checkpoint(self, ckpt_dir):
        self.freq_emb.load_state_dict(torch.load(f'{ckpt_dir}/freq_emb.pt', weights_only=True))
        self.horizon_ff_layer.load_state_dict(torch.load(f'{ckpt_dir}/horizon_ff_layer.pt', weights_only=True))
        self.input_ff_layer.load_state_dict(torch.load(f'{ckpt_dir}/input_ff_layer.pt', weights_only=True))
        self.stacked_transformer_layer.load_state_dict(torch.load(f'{ckpt_dir}/stack_transformer.pt', weights_only=True))

    def forward(self, input_ts):
        """
        input_ts: a tensor of shape (bs, context_len)
        """
        bs = input_ts.shape[0]
        patched_inputs = input_ts.reshape(bs, -1, self.input_patch_len)
        patched_pads = torch.zeros_like(patched_inputs).to('cuda')
        patched_inputs, stats = forward_transform(patched_inputs)
        concat_inputs = torch.concat([patched_inputs, patched_pads], dim=-1)

        model_input = self.input_ff_layer(concat_inputs)
        position_emb = self.position_emb(seq_length=model_input.shape[1]).to('cuda').expand(model_input.shape[0], -1, -1)
        model_input = model_input + position_emb
        f_emb = self.freq_emb(torch.zeros((bs, 1), dtype=torch.long)).to('cuda') # freq set to zero, change if needed
        model_input = model_input + f_emb
        mask = create_causal_mask(model_input.shape[0], model_input.shape[1]).to('cuda')
        model_output = self.stacked_transformer_layer(model_input, mask=mask)
        output_ts = self.horizon_ff_layer(model_output)
        output_ts = output_ts.reshape(bs, -1, self.horizon_len, self.num_outputs)
        output_ts = reverse_transform(output_ts, stats)
        return output_ts
