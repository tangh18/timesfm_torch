import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embedding(nn.Module):
    """A simple embedding layer that performs embedding lookups from ids in PyTorch.

    Attributes:
        num_classes: Number of tokens in the vocabulary.
        input_dims: Depth of the embedding output.
        scale_sqrt_depth: If set to True, activations are scaled with sqrt(embedding_dim).
        set_nan_for_oob_id: If set to True, embeddings for out-of-bound ids are set to NaN.
    """
    def __init__(self, num_classes, input_dims, scale_sqrt_depth=False, set_nan_for_oob_id=False):
        super(Embedding, self).__init__()
        assert num_classes > 0 and input_dims > 0, "num_classes and input_dims must be positive"
        self.num_classes = num_classes
        self.input_dims = input_dims
        self.scale_sqrt_depth = scale_sqrt_depth
        self.set_nan_for_oob_id = set_nan_for_oob_id
        
        # Create the embedding variable
        self.emb_var = nn.Embedding(num_classes, input_dims, padding_idx=None)

    def forward(self, ids):
        """Perform the embedding lookup.

        Args:
            ids: Tensor of indices to lookup.

        Returns:
            Tensor of embeddings corresponding to the ids.
        """
        if self.set_nan_for_oob_id:
            mask = ids >= self.num_classes
            ids = torch.where(mask, torch.tensor(0, device=ids.device, dtype=ids.dtype), ids)
        
        embs = self.emb_var(ids)

        if self.set_nan_for_oob_id:
            embs = torch.where(mask.unsqueeze(-1), torch.full_like(embs, float('nan')), embs)
        
        if self.scale_sqrt_depth:
            embs *= self.input_dims ** 0.5

        return embs

class PositionalEmbedding(nn.Module):
    """Generates position embedding for a given 1-d sequence in PyTorch.

    Attributes:
        min_timescale: Start of the geometric index. Determines the periodicity of
          the added signal.
        max_timescale: End of the geometric index. Determines the frequency of the
          added signal.
        embedding_dims: Dimension of the embedding to be generated.
    """
    def __init__(self, min_timescale=1, max_timescale=10_000, embedding_dims=0):
        super(PositionalEmbedding, self).__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.embedding_dims = embedding_dims

    def forward(self, seq_length=None, position=None):
        """Generates a tensor of sinusoids with different frequencies.

        Args:
          seq_length: an optional integer defining the output sequence length.
          position: optional tensor for each token in the sequence, shape [B, seq_length].

        Returns:
          Tensor of shape [B, seq_length, D] if position is specified, else [1, seq_length, D].
        """
        if position is None:
            assert seq_length is not None, "seq_length must be specified if position is not provided"
            position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
        else:
            assert position.dim() == 2, "position should have dimension 2"

        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        
        if self.embedding_dims % 2 == 1:  # Pad if dimensions are odd
            zero_padding = torch.zeros(signal.shape[0], signal.shape[1], 1, device=signal.device)
            signal = torch.cat([signal, zero_padding], dim=2)

        return signal

class LayerNorm(nn.Module):
    """Layer normalization in PyTorch.
    
    Attributes:
        dim: the number of features (last dimension of the input).
        epsilon: tiny value to guard against division by zero in normalization.
        use_scale: whether to use a learned scaling parameter.
        use_bias: whether to use a learned bias parameter.
        direct_scale: whether to apply scale directly without a +1.0.
    """
    def __init__(self, dim, epsilon=1e-6, use_scale=True, use_bias=True, direct_scale=False):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.use_bias = use_bias
        self.direct_scale = direct_scale

        if self.use_scale:
            init_scale = 1.0 if direct_scale else 0.0
            self.scale = nn.Parameter(torch.full((dim,), init_scale))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, inputs):
        """Applies layer normalization to the inputs."""
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, keepdim=True, unbiased=False)

        if self.direct_scale:
            scale = self.scale
        else:
            scale = 1.0 + self.scale

        # Normalize
        normalized_inputs = (inputs - mean) / torch.sqrt(variance + self.epsilon)

        # Apply scale and bias if they are used
        if self.use_scale:
            normalized_inputs = normalized_inputs * scale
        if self.use_bias:
            normalized_inputs = normalized_inputs + self.bias

        return normalized_inputs

class RmsNorm(nn.Module):
    """RMS normalization: https://arxiv.org/abs/1910.07467.

    Attributes:
        epsilon: Tiny value to guard against division by zero in sqrt.
        direct_scale: Whether to apply scale directly without a +1.0. Scale is
          initialized to 1.0 instead when true. This makes the weight compatible
          with the implementation in gshard/glam.
    """
    def __init__(self, dim, epsilon=1e-6, direct_scale=True, dtype=None):
        super(RmsNorm, self).__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.direct_scale = direct_scale
        self.dtype = dtype
        init_value = 1.0 if direct_scale else 0.0
        self.scale = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, inputs, paddings=None):
        """Applies RMS normalization to the inputs.

        Args:
            inputs: The inputs tensor. Shape should be [..., dim].
            paddings: unused.

        Returns:
            Output after applying RMS normalization, with the same shape as 'inputs'.
        """
        if paddings is not None:
            raise ValueError("Paddings are not used in this implementation.")
        if self.dtype:
            inputs = inputs.to(self.dtype)
        
        # Calculate the variance along the last dimension
        mean_squared = torch.mean(inputs ** 2, dim=-1, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(mean_squared + self.epsilon)
        
        if self.direct_scale:
            scale = self.scale
        else:
            scale = 1 + self.scale
        
        normed_inputs = normed_inputs * scale
        return normed_inputs

class ResidualBlock(nn.Module):
    """Simple feedforward block with residual connection.

    Attributes:
        input_dims: input dimension.
        hidden_dims: hidden dimension.
        output_dims: output dimension.
        dropout_prob: dropout probability.
    """
    def __init__(self, input_dims, hidden_dims, output_dims, dropout_prob=0):
        super(ResidualBlock, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.dropout_prob = dropout_prob

        # Layer definitions
        self.hidden_layer = nn.Linear(self.input_dims, self.hidden_dims)
        self.activation = nn.SiLU()  # Swish equivalent in PyTorch
        self.output_layer = nn.Linear(self.hidden_dims, self.output_dims)
        self.residual_layer = nn.Linear(self.input_dims, self.output_dims)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, inputs):
        hidden = self.activation(self.hidden_layer(inputs))
        output = self.output_layer(hidden)
        output = self.dropout(output)
        residual = self.residual_layer(inputs)
        return output + residual

class PerDimScale(nn.Module):
    """A layer to scale individual dimensions of the input in PyTorch."""
    def __init__(self, dim):
        super(PerDimScale, self).__init__()
        self.dim = dim
        # Initialize the scale parameters, equivalent to JAX's WeightInit.Constant(0.0)
        self.per_dim_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs):
        """Apply per-dimension scaling to the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor with shape [..., self.dim]

        Returns:
            torch.Tensor: Scaled tensor with shape [..., self.dim]
        """
        assert inputs.shape[-1] == self.dim, "The last dimension of the inputs must match self.dim"

        # Calculation of scale factor as per the JAX code, using softplus and normalization
        # 1.0 / torch.nn.functional.softplus(torch.tensor(0.0)) ≈ 1.442695041
        r_softplus_0 = 1.442695041
        scale_factor = r_softplus_0 / math.sqrt(self.dim)
        scale = scale_factor * F.softplus(self.per_dim_scale)

        # Element-wise multiplication of inputs and scale
        outputs = inputs * scale
        return outputs

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layer_norm = LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.layer_norm(inputs)
        x = F.relu(self.dropout1(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        x = x + inputs
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.scale_query = PerDimScale(self.d_k)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.post = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        _q = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        _k = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        _v = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scaled_q = self.scale_query(_q)

        # attn = torch.matmul(_q, _k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.matmul(scaled_q, _k.transpose(-2, -1))
        if mask is not None:
            # attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        before_post = torch.matmul(attn, _v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        after_post = self.post(before_post)
        return after_post

def create_causal_mask(batch_size, seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
    mask = mask.expand(batch_size, 1, seq_len, seq_len)  # (batch_size, 1, seq_len, seq_len)
    # mask = mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, 0.0)
    mask = mask.float().masked_fill(mask, -2.3819763e+38).masked_fill(~mask, 0.0)
    return mask

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, hidden_dim, d_model, dropout=dropout)
        self.layer_norm = RmsNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        x = self.layer_norm(inputs)
        x = self.attention(x, x, x, mask)
        x = self.dropout(x)
        x = x + inputs
        x = self.feed_forward(x)
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, hidden_dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x