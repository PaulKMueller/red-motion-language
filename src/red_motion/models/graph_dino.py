import torch

from torch import Tensor


def get_graph_dino_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    teacher_centers: Tensor,
    teacher_temperature: float = 0.06,
    student_temperature: float = 0.9,
    eps: float = 1e-20,
):
    """GraphDINO loss
    adapted from: https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/graphdino.py
    """
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temperature).softmax(dim = -1)
    teacher_probs = ((teacher_logits - teacher_centers) / teacher_temperature).softmax(dim = -1)
    loss = - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()
    
    return loss


def update_moving_average(
    ema_updater, 
    teacher_model, 
    student_model, 
    teacher_centers,
    previous_teacher_centers,
    teacher_centering_ema_updater,
    decay=None,
):
    for student_params, teacher_params in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_weights, weight_update = teacher_params.data, student_params.data
        teacher_params.data = ema_updater.update_average(teacher_weights, weight_update, decay=decay)
    
    new_teacher_centers = teacher_centering_ema_updater.update_average(teacher_centers, previous_teacher_centers)

    return new_teacher_centers


class ExponentialMovingAverage:
    """Exponential moving average with decay
    src: https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/graphdino.py
    """
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        assert (decay > 0.) and (decay < 1.), 'Decay must be in [0., 1.]'

    def update_average(
        self,
        previous_state: torch.Tensor,
        update: torch.Tensor,
        decay: float = None,
    ):
        if previous_state is None:
            return update
        if decay is not None:
            return previous_state * decay + (1 - decay) * update
        else:
            return previous_state * self.decay + (1 - self.decay) * update