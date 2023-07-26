from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput


class BaseMoEModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    gate_loss: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
