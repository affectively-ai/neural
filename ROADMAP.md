# Neural 2.0 Roadmap

Our mission is to build the "Transparent Brain" — a local-first, collaborative, and visually intuitive neural network engine.

## Phase 1: Foundation (Completed)

- [x] **Monorepo Structure:** `packages/engine` (Core) and `apps/web` (UI).
- [x] **WebGPU Engine:** Basic compute shader pipeline for forward propagation.
- [x] **Persistence:** Integration with `@buley/dash` (WASM SQLite + Vectors).
- [x] **Testing:** 100% coverage test suite using Bun.

## Phase 2: The Learning Brain (Completed)

Focus on implementing real-time training on the GPU.

- [x] **Backpropagation Shader:** Implement gradient descent in WGSL.
- [x] **Optimizers:** Implement SGD and Adam optimizers in WebGPU.
- [x] **Batch Processing:** Support mini-batch training for stability.
- [x] **Benchmarks:** Compare performance against TensorFlow.js and ONNX Runtime.

## Phase 3: The Visible Brain (Completed)

Focus on 3D visualization to make the "black box" transparent.

- [x] **3D Graph Renderer:** Use Three.js/WebGPU to render millions of neurons.
- [x] **Activity Heatmaps:** Visualize neuron activation in real-time.
- [x] **Interactive Lesioning:** Allow users to "cut" synapses and see impact live.

## Phase 4: The Collaborative Brain (Completed)

Focus on multi-user and multi-device capabilities.

- [x] **Model Export:** Export connectome to JSON.
- [x] **Model Zoo:** Public gallery of user-created architectures.
- [x] **CRDT Sync:** Basic mocked synchronization.

## Phase 5: The Hybrid Brain (In Progress)

- [x] **Brain-Computer Interface:** WebAudio/Mic Input for real-time neural injection.
- [x] **Hybrid Inference:** Offload heavy layers to cloud GPUs.

## Long Term Vision

- **Hybrid Inference:** Offload heavy layers to cloud GPUs via MCP while keeping the core graph local.
- **Brain-Computer Interface:** (Experimental) OSC/WebSocket bridges to bio-sensors.
