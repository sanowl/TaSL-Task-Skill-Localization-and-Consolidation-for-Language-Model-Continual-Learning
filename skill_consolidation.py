import torch
import math

def static_weighted_consolidation(current_units, prev_units, gamma, threshold):
    for curr_unit, prev_unit in zip(current_units, prev_units):
        if curr_unit.importance > threshold and prev_unit.importance > threshold:
            curr_unit.module.weight.data = gamma * prev_unit.module.weight.data + (1 - gamma) * curr_unit.module.weight.data
            if curr_unit.module.bias is not None:
                curr_unit.module.bias.data = gamma * prev_unit.module.bias.data + (1 - gamma) * curr_unit.module.bias.data
        elif prev_unit.importance > threshold:
            curr_unit.module.weight.data = prev_unit.module.weight.data
            if curr_unit.module.bias is not None:
                curr_unit.module.bias.data = prev_unit.module.bias.data
        elif curr_unit.importance > threshold:
            pass  # Keep current weights
        else:
            curr_unit.module.weight.data = (prev_unit.module.weight.data + curr_unit.module.weight.data) / 2
            if curr_unit.module.bias is not None:
                curr_unit.module.bias.data = (prev_unit.module.bias.data + curr_unit.module.bias.data) / 2

def adaptive_weighted_consolidation(current_units, prev_units, tau):
    for curr_unit, prev_unit in zip(current_units, prev_units):
        weight_current = math.exp(curr_unit.importance / tau)
        weight_prev = math.exp(prev_unit.cumulative_importance / tau)
        total_weight = weight_current + weight_prev

        consolidated_weight = (
            (weight_prev * prev_unit.module.weight + weight_current * curr_unit.module.weight) /
            total_weight
        )
        curr_unit.module.weight.data.copy_(consolidated_weight)

        if curr_unit.module.bias is not None:
            consolidated_bias = (
                (weight_prev * prev_unit.module.bias + weight_current * curr_unit.module.bias) /
                total_weight
            )
            curr_unit.module.bias.data.copy_(consolidated_bias)