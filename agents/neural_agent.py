from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from game.encoders import ACTION_SIZE, STATE_SIZE, encode_action, encode_state


class QNetwork(nn.Module):
    """
    Rede que recebe o par (estado, ação) e devolve um valor Q.

    Em vez de a rede ter uma saída fixa para cada ação possível, ela avalia uma ação
    estruturada específica. Isso combina melhor com trocas, porque uma troca carrega
    parâmetros como propriedades e dinheiro.
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE + ACTION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.net(state_action).squeeze(-1)


class NeuralAgent:
    """
    Agente neural baseado em Q-learning/DQN.

    O agente escolhe a ação válida com maior Q estimado. Durante treinamento, usa
    epsilon-greedy para explorar ações aleatórias.
    """

    def __init__(
        self,
        player_id: int,
        model: Optional[QNetwork] = None,
        epsilon: float = 0.05,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.player_id = player_id
        self.model = model or QNetwork()
        self.epsilon = epsilon
        self.random = random.Random(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not valid_actions:
            return {"type": "no_action"}

        if self.random.random() < self.epsilon:
            return self.random.choice(valid_actions)

        state_vec = encode_state(state)
        state_batch = np.repeat(state_vec[None, :], len(valid_actions), axis=0)
        action_batch = np.stack([encode_action(action) for action in valid_actions], axis=0)
        state_action_batch = np.concatenate([state_batch, action_batch], axis=1)

        with torch.no_grad():
            tensor = torch.tensor(state_action_batch, dtype=torch.float32, device=self.device)
            q_values = self.model(tensor).detach().cpu().numpy()

        best_index = int(np.argmax(q_values))
        return valid_actions[best_index]

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
