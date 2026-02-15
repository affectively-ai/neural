import { Neuron, Synapse } from '../types';

export class Translator {
  // Maps Neuron logical IDs (UUIDs) to Matrix Indices (0...N)
  private idToIndex: Map<string, number> = new Map();
  private indexToId: string[] = [];

  // Converts Graph -> Dense Matrices
  flatten(
    neurons: Neuron[],
    synapses: Synapse[]
  ): {
    size: number;
    weights: Float32Array;
    biases: Float32Array;
    initialValues: Float32Array;
  } {
    const size = neurons.length;
    this.idToIndex.clear();
    this.indexToId = new Array(size);

    // 1. Map IDs
    neurons.forEach((n, i) => {
      this.idToIndex.set(n.id, i);
      this.indexToId[i] = n.id;
    });

    // 2. Prepare Biases & Initial Values
    const biases = new Float32Array(size);
    const initialValues = new Float32Array(size); // Default 0

    neurons.forEach((n, i) => {
      biases[i] = n.bias;
      // initialValues could be persisted state, but defaulting to 0 for now
    });

    // 3. Prepare Weights Matrix (N x N)
    // Flattened: Row-major or implementation specific.
    // Our shader expects: weights[row * size + col]
    // where row = target neuron, col = source neuron
    const weights = new Float32Array(size * size);

    synapses.forEach((s) => {
      const fromIdx = this.idToIndex.get(s.from_id);
      const toIdx = this.idToIndex.get(s.to_id);

      if (fromIdx !== undefined && toIdx !== undefined) {
        // target (row) = toIdx
        // source (col) = fromIdx
        const flatIndex = toIdx * size + fromIdx;
        weights[flatIndex] = s.weight;
      }
    });

    return { size, weights, biases, initialValues };
  }
}
