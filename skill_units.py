import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 0.01

    def forward(self, x):
        return self.scaling * (x @ self.lora_A @ self.lora_B)

class MatrixSkillUnit:
    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.importance = None
        self.cumulative_importance = None

class LoRASkillUnit:
    def __init__(self, name, lora_layer):
        self.name = name
        self.lora_layer = lora_layer
        self.importance = None
        self.cumulative_importance = None

def initialize_matrix_skill_units(model):
    skill_units = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            skill_units.append(MatrixSkillUnit(name, module))
    return skill_units

def initialize_lora_skill_units(model):
    skill_units = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            skill_units.append(LoRASkillUnit(name, module))
    return skill_units

def lora_orthogonality_loss(model):
    loss = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            A, B = module.lora_A, module.lora_B
            loss += torch.norm(A.T @ A - torch.eye(A.shape[1]).to(A.device))
            loss += torch.norm(B.T @ B - torch.eye(B.shape[1]).to(B.device))
    return loss