def mean_std(inputs):
    """
    inputs: a tensor of shape (bs, patch_len, patch_size)
    """
    assert len(inputs.shape) == 3
    inputs_mean = inputs.mean(dim=(1, 2))
    inputs_std = inputs.std(dim=(1, 2))
    return inputs_mean, inputs_std

def forward_transform(inputs):
    mu, sigma = mean_std(inputs)
    outputs = ((inputs - mu[:, None, None]) / sigma[:, None, None])
    return outputs, (mu, sigma)

def reverse_transform(outputs, stats):
    mu, sigma = stats
    return outputs * sigma[:, None, None, None] + mu[:, None, None, None] # maybe not working when output seq_len = 1