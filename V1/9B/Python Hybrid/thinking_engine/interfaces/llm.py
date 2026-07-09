# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.interfaces.llm
================================
Abstract contract for any backend that can generate tokens. The real
implementation is ``modeling/transformer.py``'s ``TransformerBridge`` —
this Protocol mirrors ITS ACTUAL public API exactly (verified against
the real file, not assumed), so ``thinking_engine`` can type-check and
unit-test against a stub without importing the heavy NumPy transformer
stack.

IMPORTANT — API correction history
------------------------------------
An earlier revision of this file assumed ``TransformerBridge`` exposed
``is_connected()``, ``expert_boost_to_array()``, ``build_intent_bias()``,
and ``to_generate_kwargs()``. None of these exist on the real
``TransformerBridge`` in ``transformer.py``. The REAL contract is:

    bridge.backend                    -> "numpy" | "llama.cpp"  (property)
    bridge.set_intent_boost(arr)      -> stores a np.ndarray[n_experts]
    bridge.set_intent_bias(arr)       -> stores a np.ndarray[vocab_size]
    bridge.generate(prompt_ids, ...)  -> reads the stored boost/bias
                                          internally; does NOT take them
                                          as generate() kwargs

This file has been corrected to match that reality. See
``TransformerBridgeAdapter`` below for how ``thinking_engine`` bridges
the gap: the *rest of the engine* (ExpertRouter, IntentDetector, ...)
still works with plain ``Dict[int, float]`` expert-boost values and
plain identity-token lists, exactly as before — the adapter is the
ONLY place that converts those into the ``np.ndarray`` shapes the real
bridge needs and calls ``set_intent_boost``/``set_intent_bias`` prior
to ``generate()``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMBridge(Protocol):
    """Structural contract matching the REAL ``transformer.TransformerBridge``.

    Every field/method below is verified against ``transformer.py`` —
    not assumed. Do not add methods here that aren't actually present
    on the real bridge; downstream code should go through
    ``TransformerBridgeAdapter`` instead of asking the raw bridge to do
    things it doesn't do.
    """

    @property
    def backend(self) -> str:
        """Returns "numpy" or "llama.cpp" — mirrors TransformerBridge.backend."""
        ...

    def set_intent_boost(self, boost: Any) -> None:
        """Stores a per-expert bias array (np.ndarray[n_experts]) that
        DracoTransformerV1.forward() adds directly to l1 logits. Ignored
        entirely on the llama.cpp backend path (only NumPy backend
        reads self._intent_boost inside generate())."""
        ...

    def set_intent_bias(self, bias: Any) -> None:
        """Stores a per-token bias array (np.ndarray[vocab_size]).
        On the NumPy backend this is passed straight through to
        DracoTransformerV1.forward()'s ``intent_bias`` param. On the
        llama.cpp backend, TransformerBridge internally converts it via
        ``_boost_to_logit_bias()`` (top-200-by-magnitude, non-zero
        entries only) into llama.cpp's ``logit_bias`` dict format."""
        ...

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> List[int]:
        """Generate new token IDs given a prompt. Reads any previously
        set_intent_boost/set_intent_bias internally — do NOT pass
        intent_boost/intent_bias as kwargs here, the real bridge's
        generate() signature does not accept them directly."""
        ...


class TransformerBridgeAdapter:
    """Adapts thinking_engine's plain-Python engine outputs (Dict[int,
    float] expert boosts, List[int] identity token ids) into the
    np.ndarray shapes the REAL TransformerBridge needs, and calls it
    with the REAL generate() parameter names.

    This is the ONLY class in thinking_engine that imports numpy and
    talks to the real bridge's actual method names. Every other engine
    component (ExpertRouter, PromptCompiler, ...) continues to work
    with plain dicts/lists exactly as before — nothing about the
    Infrastructure/Cognition/Verification layers needed to change to
    accommodate the real bridge's shape; only this one adapter did.
    """

    def __init__(self, bridge: Any, n_experts: int = 8, vocab_size: int = 152064) -> None:
        self._bridge = bridge
        self.n_experts = n_experts
        self.vocab_size = vocab_size

    def is_connected(self) -> bool:
        """thinking_engine-side convenience — NOT a real TransformerBridge
        method. True only when a bridge is present AND it reports a real
        backend ("numpy" or "llama.cpp") rather than the stub's "stub"
        placeholder — so callers correctly treat "StubLLMBridge with no
        numpy_model attached" the same as "no bridge at all" instead of
        attempting real generation against it."""
        if self._bridge is None:
            return False
        backend = getattr(self._bridge, "backend", None)
        return backend in ("numpy", "llama.cpp")

    @property
    def backend(self) -> Optional[str]:
        if self._bridge is None:
            return None
        return self._bridge.backend

    def expert_boost_to_array(self, boost_dict: Dict[int, float]):
        """Converts an engine-native {expert_id: weight} dict into the
        np.ndarray[n_experts] shape DracoTransformerV1.forward()'s
        intent_boost param expects."""
        import numpy as np

        arr = np.zeros(self.n_experts, dtype=np.float32)
        for eid, w in boost_dict.items():
            if 0 <= eid < self.n_experts:
                arr[eid] = float(w)
        return arr

    def build_intent_bias(self, identity_token_ids: Optional[List[int]] = None, boost: float = 2.0):
        """Converts a list of identity token ids into the
        np.ndarray[vocab_size] shape intent_bias expects. Mirrors
        DracoTransformerV1.set_identity_bias()'s own construction logic
        so behaviour stays identical whether the bias is baked in via
        set_identity_bias() (persistent) or passed per-call via
        intent_bias (transient, per-request)."""
        import numpy as np

        bias = np.zeros(self.vocab_size, dtype=np.float32)
        if identity_token_ids:
            for tid in identity_token_ids:
                if 0 <= tid < self.vocab_size:
                    bias[tid] = boost
        return bias

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 256, **kwargs: Any) -> List[int]:
        """Direct pass-through to the real bridge's generate() for the
        many call sites across reasoning/planning/memory that only need
        a plain completion (e.g. one candidate "thought" in
        TreeOfThoughts, one council-member opinion in MultiAgentDebate)
        and don't need expert-boost/intent-bias routing — those go
        through generate_from_engine_output() instead. Every kwarg the
        real TransformerBridge.generate() accepts (temp, top_p, eos_ids,
        use_speculative_tree, ...) can still be passed through here."""
        if self._bridge is None:
            return []
        return self._bridge.generate(prompt_ids, max_new_tokens=max_new_tokens, **kwargs)

    def generate_from_engine_output(
        self,
        prompt_ids: List[int],
        engine_out: Dict[str, Any],
        identity_token_ids: Optional[List[int]] = None,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        min_p: float = 0.05,
        use_mirostat: bool = True,
        use_speculative: bool = True,
        adaptive_temp: bool = False,
        stream_cb: Optional[Callable[[int, float], None]] = None,
        **extra_real_bridge_kwargs: Any,
    ) -> List[int]:
        """The single call site thinking_engine.execution should use
        instead of calling bridge.generate() directly. Handles the real
        bridge's actual two-step contract:

            1. bridge.set_intent_boost(arr) / set_intent_bias(arr)
            2. bridge.generate(prompt_ids, temp=..., top_p=..., ...)
               — NOT generate(prompt_ids, intent_boost=..., ...)

        ``extra_real_bridge_kwargs`` passes straight through to the real
        generate() for parameters this adapter doesn't compute itself —
        eos_ids, use_speculative_tree, spec_tree_width/depth,
        deterministic, rep_alpha, temp_inertia, snap_delta_threshold,
        debug, profiler, stop_event, checkpoint_every, checkpoint_path,
        wal — all present on the real TransformerBridge.generate() /
        DracoTransformerV1.generate() signature.
        """
        if self._bridge is None:
            return []

        boost_arr = self.expert_boost_to_array(engine_out.get("expert_boost", {}))
        bias_arr = self.build_intent_bias(identity_token_ids)
        self._bridge.set_intent_boost(boost_arr)
        self._bridge.set_intent_bias(bias_arr)

        creativity = float(engine_out.get("creativity", 0.6))
        temp = 0.3 + creativity * 1.2

        return self._bridge.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_p=top_p,
            min_p=min_p,
            use_mirostat=use_mirostat,
            use_speculative=use_speculative,
            adaptive_temp=adaptive_temp,
            stream_cb=stream_cb,
            **extra_real_bridge_kwargs,
        )


class StubLLMBridge:
    """Zero-dependency stand-in used when no real model is attached —
    e.g. running thinking_engine's own test suite / CI without the
    heavy NumPy transformer stack installed. Exposes the SAME surface
    TransformerBridgeAdapter expects (backend property,
    set_intent_boost/set_intent_bias, generate()) so the adapter works
    identically against either the stub or the real bridge.
    """

    def __init__(
        self,
        n_experts: int = 8,
        vocab_size: int = 152064,
        numpy_model: Any = None,
        gguf_path: Optional[str] = None,
        n_gpu_layers: int = 0,
    ) -> None:
        self.n_experts = n_experts
        self.vocab_size = vocab_size
        self._numpy_model = numpy_model
        self._intent_boost: Any = None
        self._intent_bias: Any = None
        if gguf_path is not None:
            raise ImportError(
                "modeling/transformer.py not found. Install the real "
                "DracoAI transformer package to use the GGUF/llama.cpp backend."
            )

    @property
    def backend(self) -> str:
        return "numpy" if self._numpy_model is not None else "stub"

    def set_intent_boost(self, boost: Any) -> None:
        self._intent_boost = boost

    def set_intent_bias(self, bias: Any) -> None:
        self._intent_bias = bias

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 256, **kwargs: Any) -> List[int]:
        return []


def load_default_bridge(numpy_model: Any = None) -> Any:
    """Prefer the real ``modeling.transformer.TransformerBridge``; fall
    back to the lightweight stub above when that module isn't
    importable — keeps ``thinking_engine`` usable in isolation for
    tests/CI. Callers should wrap the return value in
    ``TransformerBridgeAdapter`` before use — ``engine.py`` does this
    automatically.
    """
    try:
        from modeling.transformer import TransformerBridge  # type: ignore

        return TransformerBridge(numpy_model=numpy_model)
    except ImportError:
        return StubLLMBridge(numpy_model=numpy_model)
