# Changelog

All notable changes to this project will be documented in this file.

## [4.0.1] - 2026-01-23

### Changed

- **Sync Architecture**: Aligned with Dash 4.0.0 "Stateful Serverless" release.

## [3.0.0] - 2026-01-23

### Added

- **WebNN Integration**: Added experimental support for Web Neural Network API to leverage NPU hardware for inference.
- **WebNNEngine**: New backend implementation in `packages/engine` that uses `navigator.ml`.
- **Hybrid Compute**: `NeuralEngine` now supports dual backends (WebGPU + WebNN) and auto-detects NPU availability.
- **UI Indicators**: Added "NPU ACTIVE" / "GPU ACTIVE" status badge to the web application header.

### Changed

- **Major Version Bump**: Updated `@buley/neural` to v3.0.0 to reflect significant architectural changes.
- **Web App**: Updated `apps/web` to v1.0.0.
- **Documentation**: Updated `README.md` to include WebNN in features and tech stack.

### Notes

- WebNN features require a browser with WebNN flags enabled (e.g., Chrome Canary, Edge).
- Training operations currently fallback to WebGPU to ensure stability.
