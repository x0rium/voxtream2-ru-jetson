"""Optional TensorRT dep_former runtime for VoXtream2-RU experiments."""

from __future__ import annotations

import time
from pathlib import Path

import torch


def dep_kv_buffer_names(model: torch.nn.Module) -> tuple[str, ...]:
    suffixes = ("kv_cache.k_cache", "kv_cache.v_cache", "kv_cache.cache_pos")
    names = tuple(name for name, _ in model.named_buffers() if name.endswith(suffixes))
    if len(names) != 12:
        raise RuntimeError(f"expected 12 dep_former KV buffers, got {len(names)}")
    return names


class PreloadedTensorRTEngine:
    """Deserialize a plan before large PyTorch allocations fragment Jetson RAM."""

    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt

        self.engine_path = Path(engine_path)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        plan = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        del plan
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {self.engine_path}")


class TensorRTDepFormerStep:
    """Run depth-transformer steps with explicit TensorRT-owned KV state.

    The accepted compatibility mode keeps the two-token frame init in PyTorch.
    ``run_init`` removes that dependency.  When ``init_engine_path`` is set,
    the original two-token computation runs as one q=2 TensorRT enqueue. The
    q=1 and q=2 paths may share one dynamic engine with distinct profiles. The
    sequential q=1 fallback is retained only for the rejected A/B control.
    """

    def __init__(
        self,
        engine_path: Path,
        dep_former: torch.nn.Module,
        preloaded: PreloadedTensorRTEngine | None = None,
        init_engine_path: Path | None = None,
        preloaded_init: PreloadedTensorRTEngine | None = None,
        capture_cuda_graph: bool = False,
        run_init: bool = False,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.run_init = run_init
        self.dep_former = None if run_init else dep_former
        self.buffer_names = dep_kv_buffer_names(dep_former)
        self.tensor_names = tuple(name.replace(".", "_") for name in self.buffer_names)
        loaded = preloaded or PreloadedTensorRTEngine(self.engine_path)
        if loaded.engine_path != self.engine_path:
            raise ValueError(
                f"preloaded {loaded.engine_path}, requested {self.engine_path}"
            )
        self.runtime = loaded.runtime
        self.engine = loaded.engine
        self.init_engine_path = (
            Path(init_engine_path) if init_engine_path is not None else None
        )
        self.shared_engine = (
            self.init_engine_path is not None
            and self.init_engine_path.resolve() == self.engine_path.resolve()
        )
        if self.init_engine_path is not None:
            if self.shared_engine:
                if (
                    preloaded_init is not None
                    and preloaded_init.engine is not self.engine
                ):
                    raise ValueError(
                        "the shared q=1/q=2 plan was deserialized twice; pass "
                        "the same PreloadedTensorRTEngine for both paths"
                    )
                self.init_runtime = self.runtime
                self.init_engine = self.engine
            else:
                init_loaded = preloaded_init or PreloadedTensorRTEngine(
                    self.init_engine_path
                )
                if init_loaded.engine_path != self.init_engine_path:
                    raise ValueError(
                        f"preloaded {init_loaded.engine_path}, "
                        f"requested {self.init_engine_path}"
                    )
                self.init_runtime = init_loaded.runtime
                self.init_engine = init_loaded.engine
        else:
            self.init_runtime = None
            self.init_engine = None
        self.capture_cuda_graph = capture_cuda_graph
        self.context = self.engine.create_execution_context()
        if run_init:
            init_engine = self.init_engine or self.engine
            self.init_context = init_engine.create_execution_context()
        else:
            self.init_context = None

        buffers = dict(dep_former.named_buffers())
        initial_state = tuple(buffers[name] for name in self.buffer_names)
        # cache_pos is an arange vector, not a scalar counter.  A full runtime
        # reset must restore [0, 1, ..., max_seq_len-1]; zeroing every explicit
        # state tensor aliases all KV writes to position zero.
        self.initial_state = tuple(value.clone() for value in initial_state)
        self.batch_size = initial_state[0].shape[0]
        self.step_profile_indices: tuple[int, ...] = ()
        self.step_profile_index: int | None = None
        self.init_profile_index: int | None = None
        configuration_stream = torch.cuda.Stream()
        if self._is_dynamic(self.engine):
            self.step_profile_indices = self._matching_profiles(
                self.engine, sequence_length=1
            )
            if not self.step_profile_indices:
                raise RuntimeError("dynamic TensorRT engine has no q=1 profile")
            self.step_profile_index = self.step_profile_indices[0]
            self._configure_context(
                self.engine,
                self.context,
                profile_index=self.step_profile_index,
                sequence_length=1,
                stream=configuration_stream,
            )
        if run_init and self.init_engine is not None and self._is_dynamic(
            self.init_engine
        ):
            init_profiles = self._matching_profiles(
                self.init_engine, sequence_length=2
            )
            if not init_profiles:
                raise RuntimeError("dynamic TensorRT init engine has no q=2 profile")
            self.init_profile_index = init_profiles[0]
            self._configure_context(
                self.init_engine,
                self.init_context,
                profile_index=self.init_profile_index,
                sequence_length=2,
                stream=configuration_stream,
            )
        configuration_stream.synchronize()
        self.state_a = tuple(torch.empty_like(value) for value in initial_state)
        self.state_b = tuple(torch.empty_like(value) for value in initial_state)
        self.state = self.state_a
        self.next_state = self.state_b
        batch_size = initial_state[0].shape[0]
        self.output = torch.empty((batch_size, 1, 1024), device="cuda")
        io_names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        }
        self.acoustic_logits = (
            torch.empty(
                (batch_size, 2050), device="cuda", dtype=torch.bfloat16
            )
            if "acoustic_logits" in io_names
            else None
        )
        if run_init:
            if self.init_engine is not None:
                self.init_output = torch.empty(
                    (batch_size, 2, 1024), device="cuda"
                )
            else:
                self.init_hidden = torch.empty(
                    (batch_size, 1, 1024), device="cuda", dtype=torch.bfloat16
                )
                self.init_input_pos = torch.empty(
                    (batch_size, 1), device="cuda", dtype=torch.int64
                )
                self.init_mask = torch.empty(
                    (batch_size, 1, 16), device="cuda", dtype=torch.bool
                )
        self.step_index = 0
        self.calls = 0
        self.init_calls = 0
        self.engine_enqueues = 0
        self.frames = 0
        self.frame_initialized = False
        self.default_stream_used = False
        self.graphs: tuple[torch.cuda.CUDAGraph, ...] = ()
        self.graph_index = 0
        self.graph_capture_seconds = 0.0

    @staticmethod
    def _is_dynamic(engine) -> bool:
        return -1 in tuple(engine.get_tensor_shape("hidden"))

    @staticmethod
    def _matching_profiles(engine, sequence_length: int) -> tuple[int, ...]:
        matches = []
        for profile_index in range(engine.num_optimization_profiles):
            minimum, optimum, maximum = engine.get_tensor_profile_shape(
                "hidden", profile_index
            )
            if (
                minimum[1] <= sequence_length <= maximum[1]
                and optimum[1] == sequence_length
            ):
                matches.append(profile_index)
        return tuple(matches)

    def _configure_context(
        self,
        engine,
        context,
        profile_index: int,
        sequence_length: int,
        stream: torch.cuda.Stream,
    ) -> None:
        if profile_index >= engine.num_optimization_profiles:
            raise RuntimeError(
                f"TensorRT engine has {engine.num_optimization_profiles} profiles; "
                f"cannot select profile {profile_index} for q={sequence_length}"
            )
        if context.active_optimization_profile != profile_index:
            if not context.set_optimization_profile_async(
                profile_index, stream.cuda_stream
            ):
                raise RuntimeError(
                    f"failed to select TensorRT profile {profile_index} "
                    f"for q={sequence_length}"
                )
        shapes = {
            "hidden": (self.batch_size, sequence_length, 1024),
            "input_pos": (self.batch_size, sequence_length),
            "mask": (self.batch_size, sequence_length, 16),
        }
        for name, shape in shapes.items():
            if not context.set_input_shape(name, shape):
                raise RuntimeError(
                    f"failed to set TensorRT input shape {name}={shape} "
                    f"for profile {profile_index}"
                )

    def _copy_initial_state(self) -> None:
        if self.dep_former is None:
            raise RuntimeError("PyTorch dep_former state is not available")
        buffers = dict(self.dep_former.named_buffers())
        for name, target in zip(self.buffer_names, self.state):
            target.copy_(buffers[name])

    def _begin_frame(self) -> None:
        if self.graphs and self.graph_index != 0:
            raise RuntimeError("TensorRT CUDA Graph state parity crossed a frame boundary")
        self._copy_initial_state()
        self.frames += 1

    def reset_caches(self) -> None:
        """Reset the TensorRT-owned cache to a stable A->B frame layout."""
        if not self.run_init:
            return
        self.state = self.state_a
        self.next_state = self.state_b
        for initial, value in zip(self.initial_state, self.state):
            value.copy_(initial)
        self.step_index = 0
        self.graph_index = 0
        self.frame_initialized = False

    @staticmethod
    def _bind(
        context,
        bindings: dict[str, torch.Tensor],
    ) -> None:
        for name, tensor in bindings.items():
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT tensor {name}")

    def _initialize_cuda_graph(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        started = time.perf_counter()
        saved_state = (
            tuple(value.clone() for value in self.state) if self.run_init else None
        )
        self.static_hidden = torch.empty_like(hidden)
        self.static_input_pos = torch.empty_like(input_pos)
        self.static_mask = torch.empty_like(mask)
        self.static_hidden.copy_(hidden)
        self.static_input_pos.copy_(input_pos)
        self.static_mask.copy_(mask)

        capture_stream = torch.cuda.Stream()
        second_context = self.engine.create_execution_context()
        if second_context is None:
            raise RuntimeError("failed to create second TensorRT execution context")
        if self.step_profile_index is not None:
            graph_profiles = tuple(
                profile_index
                for profile_index in self.step_profile_indices
                if profile_index != self.step_profile_index
            )
            if not graph_profiles:
                raise RuntimeError(
                    "CUDA Graph capture for the unified dynamic engine requires "
                    "profiles q=1,q=1,q=2; rebuild with "
                    "--sequence-profiles 1 1 2"
                )
            second_profile_index = graph_profiles[0]
            self._configure_context(
                self.engine,
                second_context,
                profile_index=second_profile_index,
                sequence_length=1,
                stream=capture_stream,
            )
            capture_stream.synchronize()
        contexts = (self.context, second_context)
        graphs: list[torch.cuda.CUDAGraph] = []
        state_pairs = (
            (self.state, self.next_state),
            (self.next_state, self.state),
        )

        # TensorRT requires one enqueue after any shape/address setup and before
        # capture. Each graph owns one fixed A->B or B->A KV-cache layout.
        torch.cuda.synchronize()
        for context, (state, next_state) in zip(contexts, state_pairs):
            bindings = {
                "hidden": self.static_hidden,
                "input_pos": self.static_input_pos,
                "mask": self.static_mask,
                "output": self.output,
            }
            if self.acoustic_logits is not None:
                bindings["acoustic_logits"] = self.acoustic_logits
            bindings.update(zip(self.tensor_names, state))
            bindings.update(
                (f"next_{name}", value)
                for name, value in zip(self.tensor_names, next_state)
            )
            self._bind(context, bindings)
            with torch.cuda.stream(capture_stream):
                if not context.execute_async_v3(capture_stream.cuda_stream):
                    raise RuntimeError("TensorRT CUDA Graph warm-up enqueue failed")
            capture_stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                if not context.execute_async_v3(capture_stream.cuda_stream):
                    raise RuntimeError("TensorRT CUDA Graph capture enqueue failed")
            capture_stream.synchronize()
            graphs.append(graph)

        # Warm-up and capture advanced the cache contents. Restore the true
        # state produced by this frame's two-token init.
        if saved_state is None:
            self._copy_initial_state()
        else:
            for source, target in zip(saved_state, self.state):
                target.copy_(source)
        self.contexts = contexts
        self.graphs = tuple(graphs)
        self.graph_capture_seconds = time.perf_counter() - started

    def _enqueue(
        self,
        context,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
        stream: torch.cuda.Stream,
        output: torch.Tensor | None = None,
    ) -> None:
        output = self.output if output is None else output
        bindings = {
            "hidden": hidden,
            "input_pos": input_pos,
            "mask": mask,
            "output": output,
        }
        if self.acoustic_logits is not None:
            bindings["acoustic_logits"] = self.acoustic_logits
        bindings.update(zip(self.tensor_names, self.state))
        bindings.update(
            (f"next_{name}", value)
            for name, value in zip(self.tensor_names, self.next_state)
        )
        self._bind(context, bindings)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self.state, self.next_state = self.next_state, self.state
        self.engine_enqueues += 1

    def _run_two_token_init(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape[1] != 2 or input_pos.shape[1] != 2 or mask.shape[1] != 2:
            raise ValueError("TensorRT dep_former init expects sequence length 2")
        if self.frame_initialized:
            raise RuntimeError("dep_former init called twice without reset_caches()")
        if self.init_context is None:
            raise RuntimeError("TensorRT dep_former init context is unavailable")

        current_stream = torch.cuda.current_stream()
        self.default_stream_used |= current_stream == torch.cuda.default_stream()
        if self.init_engine is not None:
            self._enqueue(
                self.init_context,
                hidden,
                input_pos,
                mask,
                current_stream,
                output=self.init_output,
            )
            self.frames += 1
            self.init_calls += 1
            self.frame_initialized = True
            return self.init_output

        for index in range(2):
            # Slices along sequence are not contiguous across the batch. Copy
            # into fixed one-token buffers expected by the static TensorRT plan.
            self.init_hidden.copy_(hidden[:, index : index + 1, :])
            self.init_input_pos.copy_(input_pos[:, index : index + 1])
            self.init_mask.copy_(mask[:, index : index + 1, :])
            self._enqueue(
                self.init_context,
                self.init_hidden,
                self.init_input_pos,
                self.init_mask,
                current_stream,
            )

        self.frames += 1
        self.init_calls += 1
        self.frame_initialized = True
        return self.output

    def __call__(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.run_init and hidden.shape[1] == 2:
            return self._run_two_token_init(hidden, input_pos, mask)
        if hidden.shape[1] != 1:
            raise ValueError(
                f"TensorRT dep_former step expects sequence length 1, got {hidden.shape[1]}"
            )
        if self.step_index == 0:
            if self.run_init:
                if not self.frame_initialized:
                    raise RuntimeError("one-token dep step called before frame init")
            else:
                # generate_frame() has just reset the PyTorch cache and executed
                # the two-token _dep_former_init, so this is the state for pos=2.
                self._begin_frame()

        current_stream = torch.cuda.current_stream()
        self.default_stream_used |= current_stream == torch.cuda.default_stream()

        if self.capture_cuda_graph:
            if not self.graphs:
                self._initialize_cuda_graph(hidden, input_pos, mask)
            self.static_hidden.copy_(hidden)
            self.static_input_pos.copy_(input_pos)
            self.static_mask.copy_(mask)
            self.graphs[self.graph_index].replay()
            self.graph_index ^= 1
        else:
            # This loop is strictly autoregressive: the preceding PyTorch op
            # produces `hidden`, and the following op consumes our output.
            self._enqueue(self.context, hidden, input_pos, mask, current_stream)

        if self.graphs:
            self.state, self.next_state = self.next_state, self.state
            self.engine_enqueues += 1
        self.step_index = (self.step_index + 1) % 14
        self.calls += 1
        return self.output

    def metrics(self) -> dict[str, object]:
        engine_bytes = self.engine_path.stat().st_size
        init_engine_file_bytes = (
            self.init_engine_path.stat().st_size
            if self.init_engine_path is not None
            else 0
        )
        return {
            "engine": str(self.engine_path),
            "engine_bytes": engine_bytes,
            "init_engine": (
                str(self.init_engine_path)
                if self.init_engine_path is not None
                else None
            ),
            "init_engine_bytes": init_engine_file_bytes,
            "shared_engine": self.shared_engine,
            "unique_plan_bytes": (
                engine_bytes
                if self.shared_engine
                else engine_bytes + init_engine_file_bytes
            ),
            "step_profile_indices": list(self.step_profile_indices),
            "step_profile_index": self.step_profile_index,
            "init_profile_index": self.init_profile_index,
            "init_mode": (
                "tensorrt_q2"
                if self.init_engine is not None
                else ("sequential_q1" if self.run_init else "pytorch_q2")
            ),
            "calls": self.calls,
            "init_calls": self.init_calls,
            "engine_enqueues": self.engine_enqueues,
            "frames": self.frames,
            "run_init": self.run_init,
            "pytorch_weights_retained": self.dep_former is not None,
            "cuda_graph": self.capture_cuda_graph,
            "cuda_graphs": len(self.graphs),
            "cuda_graph_capture_seconds": round(self.graph_capture_seconds, 3),
            "stream": "pytorch_current",
            "default_stream_used": self.default_stream_used,
            "fused_acoustic_head": self.acoustic_logits is not None,
        }


class TensorRTDepFormerFacade(torch.nn.Module):
    """Weightless module preserving the Model.dep_former cache interface."""

    def __init__(self, runtime: TensorRTDepFormerStep) -> None:
        super().__init__()
        self.runtime = runtime

    def reset_caches(self) -> None:
        self.runtime.reset_caches()

    def caches_are_enabled(self) -> bool:
        return True

    @property
    def acoustic_logits(self) -> torch.Tensor | None:
        return self.runtime.acoustic_logits

    def forward(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.runtime(hidden, input_pos, mask)
