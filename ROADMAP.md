# Neural 2.0 Roadmap

Our mission is to build the "Transparent Brain" — a local-first, collaborative, and visually intuitive neural network engine.

## Phase 1: Foundation (Completed) ✅

- [x] **Monorepo Structure:** `packages/engine` (Core) and `apps/web` (UI).
- [x] **WebGPU Engine:** Basic compute shader pipeline for forward propagation.
- [x] **Persistence:** Integration with `@buley/dash` (WASM SQLite + Vectors).
- [x] **Testing:** 100% coverage test suite using Bun.

## Phase 2: The Learning Brain (Q2 2026) 🚧

Focus on implementing real-time training on the GPU.

- [ ] **Backpropagation Shader:** Implement gradient descent in WGSL.
- [ ] **Optimizers:** Implement SGD and Adam optimizers in WebGPU.
- [ ] **Batch Processing:** Support mini-batch training for stability.
- [ ] **Benchmarks:** Compare performance against TensorFlow.js and ONNX Runtime.

## Phase 3: The Visible Brain (Q3 2026) 👁️

Focus on 3D visualization to make the "black box" transparent.

- [ ] **3D Graph Renderer:** Use Three.js/WebGPU to render millions of neurons.
- [ ] **Activity Heatmaps:** Visualize neuron activation in real-time.
- [ ] **Interactive Lesioning:** Allow users to "cut" synapses and see impact live.

## Phase 4: The Collaborative Brain (Q4 2026) 🤝

Focus on multi-user and multi-device capabilities.

- [ ] **CRDT Sync:** Leverage Dash's roadmap features to sync neural graphs between devices.
- [ ] **Model Export:** Export connectome to JSON/ONNX.
- [ ] **Model Zoo:** Public gallery of user-created architectures.

## Long Term Vision

- **Hybrid Inference:** Offload heavy layers to cloud GPUs via MCP while keeping the core graph local.
- **Brain-Computer Interface:** (Experimental) OSC/WebSocket bridges to bio-sensors.
