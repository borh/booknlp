import torch
from torch.nn import Module

def load_state_dict_compatible(model: Module, checkpoint_path: str, map_location=None) -> None:
    """
    Minimal compatibility loader: unwrap common wrappers, drop the HF positional buffer
    ("bert.embeddings.position_ids" and its "module." prefixed variant) and call
    model.load_state_dict(..., strict=False).
    """
    state = torch.load(checkpoint_path, map_location=map_location)

    # unwrap common wrapper if present (keep minimal)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(state)}")

    # Drop the HF positional buffer and its DataParallel variant which newer HF versions do not register
    state.pop("bert.embeddings.position_ids", None)
    state.pop("module.bert.embeddings.position_ids", None)

    model.load_state_dict(state, strict=False)
