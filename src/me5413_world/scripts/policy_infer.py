#!/usr/bin/env python3
"""
Model-side policy wrapper for RL local control.
Supports:
  - backend=torchscript: load a TorchScript model (.pt/.ts)
  - backend=onnx: load an ONNX model (.onnx) with onnxruntime
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Sequence


class PolicyInfer:
    def __init__(
        self,
        backend: str = "torchscript",
        model_path: str = "",
        max_lin: float = 0.6,
        max_ang: float = 1.2,
        frame_stack: int = 1,
    ):
        self.backend = (backend or "torchscript").strip().lower()
        self.model_path = model_path
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.frame_stack = int(frame_stack)
        self.obs_order = ["dist_goal", "yaw_err", "front", "left", "right", "v_curr", "w_curr"]

        self._engine = None
        self._session = None
        self._input_name = None

        if self.backend == "torchscript":
            self._load_torchscript()
        elif self.backend == "onnx":
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @property
    def ready(self) -> bool:
        return self._engine is not None or self._session is not None

    def _load_torchscript(self) -> None:
        if not self.model_path:
            return
        if not os.path.isfile(self.model_path):
            return
        try:
            import torch  # type: ignore

            self._engine = torch.jit.load(self.model_path, map_location="cpu")
            self._engine.eval()
        except Exception:
            self._engine = None

    def _load_onnx(self) -> None:
        if not self.model_path:
            return
        if not os.path.isfile(self.model_path):
            return
        try:
            import onnxruntime as ort  # type: ignore

            self._session = ort.InferenceSession(self.model_path)
            self._input_name = self._session.get_inputs()[0].name
        except Exception:
            self._session = None
            self._input_name = None

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _build_base_vector(self, obs: Dict[str, float]):
        # Keep a stable order for model input
        return [
            float(obs.get("dist_goal", 0.0)),
            float(obs.get("yaw_err", 0.0)),
            float(obs.get("front", 10.0)),
            float(obs.get("left", 10.0)),
            float(obs.get("right", 10.0)),
            float(obs.get("v_curr", 0.0)),
            float(obs.get("w_curr", 0.0)),
        ]

    def _to_vector(self, obs) -> Sequence[float]:
        if isinstance(obs, dict):
            return self._build_base_vector(obs)
        return obs

    def predict(self, obs) -> Optional[Tuple[float, float]]:
        if self.backend == "torchscript":
            if self._engine is None:
                return None
            try:
                import torch  # type: ignore

                x = torch.tensor([self._to_vector(obs)], dtype=torch.float32)
                y = self._engine(x)
                out = y.detach().cpu().numpy().reshape(-1)
                if out.shape[0] < 2:
                    return None
                v = self._clamp(float(out[0]), -self.max_lin, self.max_lin)
                w = self._clamp(float(out[1]), -self.max_ang, self.max_ang)
                return v, w
            except Exception:
                return None

        if self.backend == "onnx":
            if self._session is None or self._input_name is None:
                return None
            try:
                import numpy as np  # type: ignore

                x = np.array([self._to_vector(obs)], dtype=np.float32)
                out = self._session.run(None, {self._input_name: x})[0].reshape(-1)
                if out.shape[0] < 2:
                    return None
                v = self._clamp(float(out[0]), -self.max_lin, self.max_lin)
                w = self._clamp(float(out[1]), -self.max_ang, self.max_ang)
                return v, w
            except Exception:
                return None

        return None
