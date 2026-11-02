"""Train replay with state snapshots via the official debug accessor only."""
from copy import deepcopy

from lifemate.tools.env_adapter import ToolEnvAdapter


class TrainReplaySession(ToolEnvAdapter):
    """Evaluation-only session. No get_state API is added to the agent adapter.

    Build a full expected state from the initial state plus label.final_state,
    then compare after calls. This also checks apps omitted from the label stay
    unchanged. No snapshot is returned to callers or written to reports.
    """
    def __init__(self, data_root, sample_id, **kwargs):
        super().__init__(data_root, "train", sample_id, **kwargs)
        self._initial = deepcopy(self._invoke("get_state"))

    def compare_final_state(self, reference: dict) -> dict:
        with self._lock:
            self._active()
            if not isinstance(reference, dict):
                raise ValueError("reference final_state must be an object")
            actual = self._invoke("get_state")
            expected = deepcopy(self._initial)
            expected.update(deepcopy(reference))
            changed_matches = sum(app in actual and actual[app] == value
                                  for app, value in reference.items())
            return {"full_state_match": actual == expected,
                    "label_apps": len(reference), "label_apps_matched": changed_matches}
