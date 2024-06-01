import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, num_class, device):
        super(Model, self).__init__()
        self.FeatureExtraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.SequenceModeling = nn.LSTM(64, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.Prediction = nn.Linear(256, num_class)
        self.device = device

    def forward(self, input):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = visual_feature.view(visual_feature.size(0), visual_feature.size(1), -1)
        visual_feature = visual_feature.permute(0, 2, 1)
        contextual_feature, _ = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous().view(-1, contextual_feature.shape[2]))
        return prediction.view(contextual_feature.size(0), -1, prediction.shape[-1])
