"""Fixed-size spatial blocks and their reusable CUDA capture."""

import torch

SPATIAL_KEYS = ("cells", "cell_mask", "neighbors", "globals", "previous")


class StaticSpatialGraph:
    """Replay one no-grad block of a default-compiled spatial forward.

    The graph reads live parameter storage. Adam, rollback and archive weight
    switches copy into that storage. The owner must rebuild the runner if it
    replaces storage or reconfigures the compiled computation. Other streams
    and input signatures use the same forward without touching static buffers.
    """

    def __init__(self, runner, rows, cells):
        self._runner = runner
        self._shape = (rows, cells)
        self._capture = None

    def __call__(self, part):
        cells = part["cells"]
        if (
            torch.is_grad_enabled()
            or cells.device.type != "cuda"
            or cells.shape[:2] != self._shape
        ):
            return self._runner(part)
        # Cloning a strided view can change its layout and the chosen kernels.
        if self._capture is None and not all(tensor.is_contiguous() for tensor in part.values()):
            return self._runner(part)
        current = torch.cuda.current_stream(cells.device)
        signature = (
            current,
            torch.is_inference_mode_enabled(),
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            tuple(
                (key, value.shape, value.stride(), value.dtype)
                for key, value in part.items()
            ),
        )
        if self._capture is None:
            self._capture = self._capture_block(part, current, signature)
        recorded, graph, inputs, (cells_out, context) = self._capture
        if signature != recorded:
            return self._runner(part)
        torch._foreach_copy_(list(inputs.values()), [part[key] for key in inputs])
        graph.replay()
        # Callers may retain multiple blocks until concatenation.
        return cells_out.clone(), context.clone()

    def _capture_block(self, part, current, signature):
        inputs = {key: value.clone() for key, value in part.items()}
        stream = torch.cuda.Stream(device=current.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._runner(inputs)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self._runner(inputs)
        return signature, graph, inputs, outputs


def iter_spatial_blocks(batch, size):
    for start in range(0, len(batch["cells"]), size):
        keys = SPATIAL_KEYS + (("unit_indices",) if "unit_indices" in batch else ())
        part = {key: batch[key][start : start + size] for key in keys}
        count = len(part["cells"])
        if count < size:
            part = {
                key: torch.cat(
                    (value, value[:1].expand(size - count, *value.shape[1:])), 0
                )
                for key, value in part.items()
            }
        yield part, count
