import torch
import torch.nn as nn
import torch.nn.functional as F

class SubjectEncoder(nn.Module):
    def __init__(self, n_subjects: int = 1, n_filters: int = 16):
        super(SubjectEncoder, self).__init__()
        self.n_subjects = n_subjects
        self.fn1 = nn.Linear(n_subjects, n_filters)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(x, num_classes=self.n_subjects).to(torch.float32)
        x = self.fn1(x)
        x = self.act(x)
        return x