# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-28

### Added

- Initial release: framework-agnostic implementation of ReasoningBank
  (Ouyang et al., ICLR 2026).
- Neutral `Turn` / `Trajectory` types with coercion from `list[dict]` or
  duck-typed message objects.
- `MemoryItem` model with `signature_text()`.
- `MemoryStore` protocol with `InMemoryStore` and `SQLiteVecStore` (sqlite-vec
  backed, cosine similarity) implementations.
- `Embedder` protocol with `MiniLMEmbedder` default; PolyRT-hosted embedder stub.
- LLM-as-judge, induction (success/failure distillation), retrieval, and
  merge / link / replace integration.
- `ReasoningBank` public API: `retrieve`, `ingest_trajectory`, `add_manual`,
  `list`, `delete`, `export`, `import_`, `format_as_system_block`.
- `matts_run` — Memory-Aware Test-Time Scaling as a functional, framework-agnostic
  submodule that takes a caller-supplied `rollout_fn`. Handles truncated /
  invalid contrast-LLM JSON by warning and returning an empty memory list
  rather than crashing.
