import torch

class ImportanceCalculator:
    def __init__(self, alpha1=0.85, alpha2=0.85):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.smoothed_sensitivity = {}
        self.uncertainty = {}

    def compute_first_order_importance(self, model, loss):
        importance = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad = param.grad
                importance[name] = (grad * param).abs().detach()
        return importance

    def compute_second_order_importance(self, model, loss):
        importance = {}
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        for (name, param), grad in zip(model.named_parameters(), grads):
            fisher = grad.pow(2).detach()
            importance[name] = param.abs() * fisher - 0.5 * fisher * param.pow(2)
        return importance

    def update_importance(self, name, current_importance):
        if name not in self.smoothed_sensitivity:
            self.smoothed_sensitivity[name] = current_importance
            self.uncertainty[name] = torch.zeros_like(current_importance)
        else:
            self.smoothed_sensitivity[name] = (
                self.alpha1 * self.smoothed_sensitivity[name] +
                (1 - self.alpha1) * current_importance
            )
            self.uncertainty[name] = (
                self.alpha2 * self.uncertainty[name] +
                (1 - self.alpha2) * abs(current_importance - self.smoothed_sensitivity[name])
            )
        return self.smoothed_sensitivity[name] * self.uncertainty[name]