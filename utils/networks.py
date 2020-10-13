from utils.obs_dict import ObsDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, set_final_bias):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

        if set_final_bias:
            self.fc3.weight.data.mul_(0.1)
            self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        if type(x) is ObsDict:
            x = x['obs_vec']

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class ConvNetwork(nn.Module):
    """
    Convolutional Network architecture for Pommerman
    Based on: https://arxiv.org/pdf/1812.07297.pdf
    """

    def __init__(self, input_size, num_input_channels, output_vec_size, k=3, s=1, p=0):
        super(ConvNetwork, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, out_channels=16, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k, stride=s, padding=p)

        self.last_featuremaps_size = input_size - 3 * k + 3
        self.fc = nn.Linear(64 * self.last_featuremaps_size ** 2, output_vec_size)

    def forward(self, x):
        if type(x) is ObsDict:
            x = x['obs_map']

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_flat = x.view(-1, self.fc.in_features)
        out_vector = F.relu(self.fc(x_flat))
        return out_vector


class CNN_MLP_hybrid(nn.Module):
    """
    Model that contains both a ConvNetwork and a MLPNetwork
    """

    def __init__(self, input_vec_len, mlp_output_vec_len, mlp_hidden_size,
                 input_maps_size, num_input_channels, cnn_output_vec_len, set_final_bias):
        super(CNN_MLP_hybrid, self).__init__()
        self.mlp = MLPNetwork(input_vec_len + cnn_output_vec_len, mlp_output_vec_len, mlp_hidden_size,
                              set_final_bias=set_final_bias)
        self.cnn = ConvNetwork(input_maps_size, num_input_channels, cnn_output_vec_len)

    def forward(self, x):
        """
        x: a dictionary of inputs of the following names and shapes:
            'obs_vec': (batch_size, obs_vec_size)
            'obs_map': (batch_size, num_channels, width, length)
        """
        vec_out_cnn = self.cnn(x['obs_map'])
        vec_in_mlp = torch.cat([x['obs_vec'], vec_out_cnn], dim=-1)
        vec_out_mlp = self.mlp(vec_in_mlp)
        return vec_out_mlp
