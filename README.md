# Neural 2.0: The Transparent Brain

**Neural** is a local-first, WebGPU-accelerated neural graph database. It allows you to build, train, and visualize neural networks that live entirely on your device, with no cloud dependencies.

## Key Features

- **WebGPU Compute:** Runs forward/backward propagation on the GPU for high-performance training.
- **Local Persistence:** Uses **Dash** (WASM SQLite + OPFS) to store the neural graph structure.
- **Semantic Neurons:** Tag neurons with natural language descriptions (vector embeddings) for queryability.
- **Real-time Logic:** A "living" network that persists state across sessions.

## Project Structure

This is a monorepo managed by Bun namespaces:

- **`packages/engine`**: The core library. Contains the `GPUEngine`, `Translator`, and Dash database persistence layer.
- **`apps/web`**: A Next.js 14 application demonstrating the "Transparent Brain" UI.

## Getting Started

### Prerequisites

- [Bun](https://bun.sh) (Required for workspace management)
- A browser with **WebGPU** support (Chrome 113+, Edge, or Firefox Nightly).

### Development

1.  **Install Dependencies:**

    ```bash
    bun install
    ```

2.  **Run the Demo:**
    ```bash
    cd apps/web
    bun dev
    ```

## Architecture

1.  **Storage:** `neurons` and `synapses` tables in local SQLite.
2.  **Logic:** TypeScript classes (`NeuronRepository`) handle CRUD.
3.  **Compute:** `brain.wgsl` compute shaders execute the math.

## License

MIT
