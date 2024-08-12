import torch
import torch.nn as nn
import torch.nn.functional as F
from skill_units import initialize_matrix_skill_units, initialize_lora_skill_units, lora_orthogonality_loss
from importance_calculation import ImportanceCalculator
from skill_localization import compute_skill_unit_importance, update_cumulative_importance
from skill_consolidation import static_weighted_consolidation, adaptive_weighted_consolidation

class TaSL:
    def __init__(self, model, num_tasks, beta=0.7, tau=0.15, use_lora=False, use_adaptive_consolidation=True):
        self.model = model
        self.num_tasks = num_tasks
        self.beta = beta
        self.tau = tau
        self.use_lora = use_lora
        self.use_adaptive_consolidation = use_adaptive_consolidation
        self.skill_units = initialize_lora_skill_units(model) if use_lora else initialize_matrix_skill_units(model)
        self.importance_calculator = ImportanceCalculator()

    def train_task(self, task_id, dataloader, optimizer, num_epochs):
        for epoch in range(num_epochs):
            for batch in dataloader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)

                if self.use_lora:
                    loss += lora_orthogonality_loss(self.model)

                optimizer.zero_grad()
                loss.backward()

                importance = self.importance_calculator.compute_second_order_importance(self.model, loss)
                for name, imp in importance.items():
                    importance[name] = self.importance_calculator.update_importance(name, imp)

                compute_skill_unit_importance(self.skill_units, importance)

                optimizer.step()

        update_cumulative_importance(self.skill_units, self.beta)

        if task_id > 0:
            prev_model = type(self.model)()
            prev_model.load_state_dict(self.model.state_dict())
            prev_skill_units = initialize_lora_skill_units(prev_model) if self.use_lora else initialize_matrix_skill_units(prev_model)
            
            if self.use_adaptive_consolidation:
                adaptive_weighted_consolidation(self.skill_units, prev_skill_units, self.tau)
            else:
                static_weighted_consolidation(self.skill_units, prev_skill_units, gamma=0.5, threshold=0.5)

    def train_all_tasks(self, dataloaders, optimizer, num_epochs):
        for task_id, dataloader in enumerate(dataloaders):
            print(f"Training task {task_id}")
            self.train_task(task_id, dataloader, optimizer, num_epochs)

# Usage example:
# model = YourModelArchitecture()
# tasl = TaSL(model, num_tasks=5, use_lora=True, use_adaptive_consolidation=True)
# dataloaders = [task1_dataloader, task2_dataloader, ...]
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# tasl.train_all_tasks(dataloaders, optimizer, num_epochs=10)