# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small simulator-side client; requires only websockets, msgpack and NumPy."""

from cosmos_framework.inference.causal_action.wire import packb, unpackb


class CausalPolicyClient:
    def __init__(self, uri="ws://localhost:8000", *, control_mode="joint"):
        from websockets.sync.client import connect

        self.connection = connect(uri, compression=None, max_size=None, ping_interval=None)
        self.metadata = unpackb(self.connection.recv())
        if self.metadata.get("protocol") != "cosmos-causal-action-v2":
            self.close()
            raise ValueError("Expected a cosmos-causal-action-v2 server")
        if control_mode not in self.metadata["available_control_modes"]:
            self.close()
            raise ValueError(f"Server does not support control_mode={control_mode!r}")
        self.control_mode = control_mode

    def infer(self, observation):
        self.connection.send(packb(observation))
        response = unpackb(self.connection.recv())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def reset(self, *, prompt):
        return self.infer(dict(reset=True, prompt=prompt, control_mode=self.control_mode))

    def actions(self, response):
        """Select server-packed native joint/EEF actions without unified55 slot logic."""
        actions = response["action"][self.control_mode]
        if actions is None:
            raise ValueError(f"Server returned no valid {self.control_mode} actions")
        return actions

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
