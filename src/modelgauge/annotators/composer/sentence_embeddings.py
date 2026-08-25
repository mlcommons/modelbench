import hashlib
import numpy as np
from functools import lru_cache
from typing import Sequence

from modelgauge.annotators.composer.context import EvalContext
from modelgauge.annotators.composer.nodes import (
    CacheableNodeMixin,
    Enricher,
    NodeOutput,
)
from modelgauge.annotators.composer.verdict import Verdict

from evaluators.components.sentence_segmenter import Sentence


@lru_cache(maxsize=1)
def available_accelerators() -> tuple[str, ...]:
    """Return the accelerators PyTorch reports, most preferred first.

    Cached because probing costs a CUDA/MPS driver call. Tests that fake
    availability must call `clear_device_cache` around the change.
    """
    import torch

    found = []
    if torch.cuda.is_available():
        found.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        found.append("mps")
    return tuple(found)


def resolve_device(name: str | None = None) -> str:
    """Resolve a requested device to a concrete PyTorch device string.

    - `cpu` (the D2 default) always succeeds and never probes PyTorch, so a
      host with no accelerator at all runs without touching a driver.
    - `cuda`/`mps` are explicit: if PyTorch reports the device unavailable
      this raises rather than falling back, because a silent downgrade would
      hide the reason a run is slow.
    - `auto` prefers CUDA, then Apple MPS, then CPU. It is never the default;
      it makes results host-dependent, which callers must opt into.

    Resolving an already-resolved name returns it unchanged, so a caller that
    resolves once and passes the result down cannot re-resolve into a
    different device.
    """
    if name == "cpu":
        return name
    if name == "auto":
        available = available_accelerators()
        return available[0] if available else "cpu"
    if name in ("cuda", "mps"):
        if name in available_accelerators():
            return name
        raise ValueError(f"device {name!r} is not available")
    raise ValueError(
        f"unknown execution device {name!r}; choose one of "
        f"{', '.join(("cpu", "cuda", "mps", "auto"))}"
    )


class BgeEmbeddingProvider:
    EMBEDDING_DIM = 768
    DEFAULT_MAX_SEQ_LENGTH = 512

    def __init__(
        self,
        model_name: str,
        revision: str,
        allow_download: bool = False,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        # Resolved once, here, so an explicit `--device cuda` on a host
        # without CUDA fails while the run is still being set up rather than
        # after stages 1-8 have already processed every row.
        self.device = resolve_device(device)
        self.allow_download = allow_download
        self.model = None

    @lru_cache(maxsize=4)
    def _load_model(self, max_seq_length: int):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            self.model_name,
            revision=self.revision,
            local_files_only=not self.allow_download,
            device=self.device,
        )
        model.max_seq_length = max_seq_length
        return model

    def embed_sentences(
        self,
        sentences: list[str],
        batch_size: int = 32,
        max_seq_length: int | None = None,
    ) -> np.ndarray:
        # `batch_size` stays fixed across devices: minibatch size changes the
        # padding within a batch, so varying it by device would make results
        # differ for reasons unrelated to the device itself.
        if not sentences:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)
        if max_seq_length is None:
            max_seq_length = self.DEFAULT_MAX_SEQ_LENGTH
        if self.model is None:
            self.model = self._load_model(max_seq_length)
        return np.asarray(
            self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )


class SentenceEmbeddingNode(Enricher, CacheableNodeMixin):
    def __init__(
        self,
        name: str,
        routes: Sequence[str | Verdict],
        segmenter_node_name: str,
        embedding_provider=None,
        model_name: str | None = None,
        model_revision: str | None = None,
        allow_download: bool = False,
        device: str = None,
    ) -> None:
        if embedding_provider and (
            model_name or model_revision or allow_download or device
        ):
            raise ValueError(
                "Cannot specify both embedding_provider and model_name, model_revision, allow_download, or device."
            )
        self.embedding_provider = embedding_provider or BgeEmbeddingProvider(
            model_name=model_name,
            revision=model_revision,
            allow_download=allow_download,
            device=device if device else "cpu",
        )
        self.model_name = (
            model_name if model_name else self.embedding_provider.model_name
        )
        # The provider exposes `revision` (see BgeEmbeddingProvider, and the
        # `revision=` argument passed above); the node's own field is
        # `model_revision`. Reading `provider.model_revision` here raised
        # AttributeError for every injected provider.
        self.model_revision = (
            model_revision if model_revision else self.embedding_provider.revision
        )
        self.segmenter_node_name = segmenter_node_name
        super().__init__(name, routes=routes)

    def _get_sentences(self, ctx: EvalContext) -> list[Sentence]:
        sentences = ctx.ancestor_output(self.segmenter_node_name)
        assert (
            sentences is not None
        ), f"No parent output found for segmenter node name {self.segmenter_node_name}."
        return [sentence.text for sentence in sentences.value]

    def cache_key(self, ctx: EvalContext) -> str:
        sentences = self._get_sentences(ctx)
        payload = f"{self.model_name}\x00{self.model_revision}\x00{chr(0).join(sentences)}".encode(
            "utf-8"
        )
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def run(self, ctx: EvalContext) -> NodeOutput:
        sentences = self._get_sentences(ctx)
        vectors = self.embed_sentence_texts(sentences)
        return NodeOutput(value=vectors, original_ctx=ctx)

    def embed_sentence_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.embedding_provider.embed_sentences(texts))
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(
                "embedding provider returned the wrong shape for decoded sentences: "
                f"{vectors.shape!r} for {len(texts)} sentence(s)"
            )
        return vectors
