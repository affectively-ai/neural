# Neural 2.0: The Transparent Brain

**Neural** is a local-first, WebGPU-accelerated neural graph database. It allows you to build, train, and visualize neural networks that live entirely on your device, with no cloud dependencies.

![Neural 2.0 Banner](https://placehold.co/1200x400/000000/0d9488?text=Neural+2.0:+The+Transparent+Brain)

## Key Features

### The Learning Brain (Compute)

- **WebGPU Backpropagation**: Custom WGSL kernels for massively parallel forward/backward passes.
- **WebNN (NPU) Inference**: Direct access to Neural Processing Units for battery-efficient, native-speed execution.
- **Real-Time Training**: Interactive stochastic gradient descent (SGD) engine running at 60fps.
- **Batch Processing**: Supports mini-batch training with 3D compute dispatch.

### The Visible Brain (Visualization)

- **3D Interactive Connectome**: Render 10k+ neurons and synapses using **Three.js** and InstancedMesh.
- **Force-Directed Layout**: Physics-based arrangement of the neural graph in real-time.
- **Live Lesioning**: Click on synapses to cut them and see the network recompile instantly.

### The Persistent Brain (Storage)

- **Local-First**: Built on **Dash** (SQLite + OPFS), storing your neural graph privately on your device.
- **Semantic Tagging**: Neurons support vector embeddings for semantic search.

### The Collaborative Brain (Sharing)

- **Model Zoo**: Explore preset architectures (XOR, Recurrent Loops).
- **Import/Export**: Share your neural graphs via JSON.

### The Hybrid Brain (Experimental)

- **Brain-Computer Interface**: Inject real-time audio (Microphone) directly into the neural network's input neurons.

## Architecture

This project is a high-performance **Bun Monorepo**:

- **`packages/engine`**: The Core.
  - `GPUEngine`: manages WebGPU buffers & pipelines.
  - `Translator`: Flattens graph topology into dense matrices.
  - `NeuronRepository`: Persistence layer.
- **`apps/web`**: The Interface.
  - Next.js 16 App Router.
  - React Three Fiber (R3F) for visualization.
  - Glassmorphism Design System.

## Getting Started

### Prerequisites

- [Bun](https://bun.sh) v1.0+
- A WebGPU-enabled browser (Chrome 113+, Edge, Firefox Nightly).

### Installation

```bash
# Install dependencies
bun install
```

### Running the Demo

```bash
cd apps/web
bun dev
```

Open [http://localhost:3000](http://localhost:3000) to see the Transparent Brain.

### Running Benchmarks

Measure your GPU's inference and training throughput:

```bash
cd packages/engine
bun run bench
```

## Tech Stack

- **Runtime**: Bun
- **Frontend**: Next.js, TailwindCSS, React Three Fiber
- **Compute**: WebGPU (WGSL) + WebNN (NPU)
- **Storage**: @buley/dash (WASM SQLite)

## License

MIT
