import torch
from torch import Tensor, nn
from ._scalarizer_base import Scalarizer

class GradNormScalarizer(Scalarizer):
    def __init__(self, num_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.alpha = alpha
        self.register_buffer("initial_losses", None)

    def forward(self, values: Tensor, model: nn.Module = None) -> Tensor:
        if self.initial_losses is None:
            self.initial_losses = values.detach().clone()
        
        if model is not None:
            norms = self._compute_gradient_norms(values, model)
            loss_ratios = values / self.initial_losses
            target_norm = torch.mean(norms) * (loss_ratios ** self.alpha)
            self.weights.data = target_norm / norms
            
        return (values * self.weights).sum()
    
    def _compute_gradient_norms(self, values: Tensor, model: nn.Module) -> Tensor:
        norms = []
        for loss in values:
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            norm = torch.norm(torch.cat([g.view(-1) for g in grads]))
            norms.append(norm)
        return torch.stack(norms)
