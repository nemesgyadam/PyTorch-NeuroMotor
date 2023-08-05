import torch
import torch.nn as nn

class ConditionedBatchNorm(nn.Module):
    '''
    Conditional Batch Normalization: This technique is used frequently in generative models
    but can also be useful in other types of neural networks. 
    The idea is to include a batch normalization layer that is conditioned on the subject id. 
    The subject id is again one-hot encoded (or embedded) and then used to calculate the mean and variance
    for the normalization. This allows the model to learn different normalization parameters for each subject,
    which can help it adapt to subject-specific characteristics.
    '''
    def __init__(self, num_features, num_conditions):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.embed = nn.Embedding(num_conditions, 2*num_features)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialize scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma * out + beta  # Affine transform
        return out
