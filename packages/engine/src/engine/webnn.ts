/// <reference path="../webnn-types.d.ts" />

export class WebNNEngine {
  context: MLContext | null = null;
  builder: MLGraphBuilder | null = null;
  graph: MLGraph | null = null;

  // Buffers/State
  networkSize: number = 0;
  batchSize: number = 1;

  // We keep weights/biases in memory to rebuild graph if needed,
  // though for strict WebNN we bakw them into constants.
  weights: Float32Array | null = null;
  biases: Float32Array | null = null;

  isReady = false;

  async init() {
    if (!navigator.ml) {
      console.warn('WebNN: navigator.ml not supported');
      return;
    }

    try {
      // Prefer NPU, fallback to GPU if NPU not available, though we really want NPU for "cool factor"
      // Note: browser support for 'npu' deviceType is bleeding edge.
      this.context = await navigator.ml.createContext({
        deviceType: 'npu',
        powerPreference: 'low-power',
      });
      console.log('WebNN: NPU Context created');

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.builder = new (window as any).MLGraphBuilder(this.context);
      this.isReady = true;
    } catch (e) {
      console.error('WebNN Init Error (likely no NPU or flag disabled):', e);
      // Fallback? or just stay uninitialized
    }
  }

  async prepareModel(
    size: number,
    weights: Float32Array,
    biases: Float32Array,
    batchSize: number = 1
  ) {
    if (!this.context || !this.builder) return;
    this.networkSize = size;
    this.batchSize = batchSize;
    this.weights = weights;
    this.biases = biases;

    // Build the Computational Graph
    // Model: Y = Tanh((X * W) + B)
    // Shapes:
    // X: [batchSize, networkSize]
    // W: [networkSize, networkSize]
    // B: [networkSize] (Broadcasted)
    // Y: [batchSize, networkSize]

    try {
      const builder = this.builder;

      // 1. Input Operand (Variable)
      const inputDesc: MLOperandDescriptor = {
        dataType: 'float32',
        dimensions: [batchSize, size],
      };
      const input = builder.input('input', inputDesc);

      // 2. Constants (Weights & Biases)
      // Note: WebNN matmul expects typically [I, J] * [J, K] -> [I, K]
      // Our weights are N*N flattened.
      const weightDesc: MLOperandDescriptor = {
        dataType: 'float32',
        dimensions: [size, size],
      };
      // WebNN might require specific buffer types, Float32Array is good.
      const weightConstant = builder.constant(weightDesc, weights);

      const biasDesc: MLOperandDescriptor = {
        dataType: 'float32',
        dimensions: [size], // 1D, will broadcast to [batch, size]
      };
      const biasConstant = builder.constant(biasDesc, biases);

      // 3. Operations
      // MatMul: [batch, size] * [size, size] -> [batch, size]
      const matmul = builder.matmul(input, weightConstant);

      // Add Bias (Broadcast)
      const added = builder.add(matmul, biasConstant);

      // Activation
      const output = builder.tanh(added);

      // 4. Build
      this.graph = await builder.build({ output: output });
      console.log('WebNN: Graph compiled successfully');
    } catch (e) {
      console.error('WebNN Build Error:', e);
    }
  }

  async runTick(inputs: Float32Array): Promise<Float32Array> {
    if (!this.context || !this.graph) {
      throw new Error('WebNN not ready');
    }
    if (inputs.length !== this.networkSize * this.batchSize) {
      throw new Error(
        `Input size mismatch. Expected ${
          this.networkSize * this.batchSize
        }, got ${inputs.length}`
      );
    }

    const outputs = new Float32Array(this.networkSize * this.batchSize);

    const inputsMap = { input: inputs };
    const outputsMap = { output: outputs };

    await this.context.compute(this.graph, inputsMap, outputsMap);

    return outputs;
  }
}
