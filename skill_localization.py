import torch
from skill_units import MatrixSkillUnit, LoRASkillUnit

def compute_skill_unit_importance(skill_units, importance):
    for unit in skill_units:
        if isinstance(unit, MatrixSkillUnit):
            weight_importance = importance.get(f"{unit.name}.weight", None)
            bias_importance = importance.get(f"{unit.name}.bias", None)
            if weight_importance is not None:
                unit_importance = weight_importance.mean()
                if bias_importance is not None:
                    unit_importance += bias_importance.mean()
                unit.importance = unit_importance.item()
        elif isinstance(unit, LoRASkillUnit):
            a_importance = importance.get(f"{unit.name}.lora_A", None)
            b_importance = importance.get(f"{unit.name}.lora_B", None)
            if a_importance is not None and b_importance is not None:
                unit_importance = (a_importance.mean() + b_importance.mean()) / 2
                unit.importance = unit_importance.item()

def update_cumulative_importance(skill_units, beta):
    for unit in skill_units:
        if unit.cumulative_importance is None:
            unit.cumulative_importance = unit.importance
        else:
            unit.cumulative_importance = (
                beta * unit.cumulative_importance +
                (1 - beta) * unit.importance
            )