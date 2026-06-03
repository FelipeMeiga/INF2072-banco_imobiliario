from __future__ import annotations

import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from game.encoders import (
    ENCODER_RAW,
    encode_action,
    encode_state,
    get_action_size,
    get_state_size,
)

NEAT_ENCODER = ENCODER_RAW
NEAT_INPUT_SIZE = get_state_size(NEAT_ENCODER) + get_action_size(NEAT_ENCODER)


class NeatAgent:
    """
    NEAT agent that scores each valid (raw state, raw action) pair.

    The network has one output. The environment still supplies only legal
    actions, so NEAT evolves preferences over the current action set instead of
    needing a fixed output head for every possible structured action.
    """

    def __init__(
        self,
        player_id: int,
        network: Any,
        seed: Optional[int] = None,
    ):
        self.player_id = player_id
        self.network = network
        self.random = random.Random(seed)

    def choose_action(self, state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not valid_actions:
            return {"type": "no_action"}

        state_vec = encode_state(state, encoder=NEAT_ENCODER)
        best_score = float("-inf")
        best_actions: List[Dict[str, Any]] = []

        for action in valid_actions:
            action_vec = encode_action(action, encoder=NEAT_ENCODER)
            network_input = np.concatenate([state_vec, action_vec], axis=0)
            score = float(self.network.activate(network_input.tolist())[0])

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return self.random.choice(best_actions)


def create_neat_network(genome: Any, config: Any) -> Any:
    import neat

    return neat.nn.FeedForwardNetwork.create(genome, config)


def save_neat_agent_checkpoint(
    path: str,
    genome: Any,
    config: Any,
    generation: int,
    fitness: float,
):
    payload = {
        "algorithm": "neat",
        "encoder": NEAT_ENCODER,
        "input_size": NEAT_INPUT_SIZE,
        "state_size": get_state_size(NEAT_ENCODER),
        "action_size": get_action_size(NEAT_ENCODER),
        "generation": generation,
        "fitness": fitness,
        "genome": genome,
        "config": config,
    }
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def load_neat_agent_checkpoint(path: str) -> Tuple[Any, Dict[str, Any]]:
    with open(path, "rb") as file:
        payload = pickle.load(file)

    if payload.get("algorithm") != "neat":
        raise ValueError(f"Checkpoint nao e NEAT: {path}")

    input_size = int(payload.get("input_size", -1))
    if input_size != NEAT_INPUT_SIZE:
        raise ValueError(
            f"Checkpoint NEAT incompativel: input salvo={input_size}, atual={NEAT_INPUT_SIZE}"
        )

    network = create_neat_network(payload["genome"], payload["config"])
    metadata = {
        "algorithm": payload.get("algorithm"),
        "encoder": payload.get("encoder", NEAT_ENCODER),
        "generation": payload.get("generation"),
        "fitness": payload.get("fitness"),
        "input_size": input_size,
    }
    return network, metadata
