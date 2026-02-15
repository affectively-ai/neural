import shaderCode from './shaders/brain.wgsl?raw';
import trainingShaderCode from './shaders/training.wgsl?raw';
// import trainingShaderCode from './shaders/training.wgsl?raw'; // Handled in replace block above, this is safety check

export class GPUEngine {
  device: GPUDevice | null = null;
  pipeline: GPUComputePipeline | null = null;
  bindGroup: GPUBindGroup | null = null;

  // Training Buffers
  deltaBuffer: GPUBuffer | null = null;
  targetBuffer: GPUBuffer | null = null;
  paramBuffer: GPUBuffer | null = null;

  trainingPipeline: GPUComputePipeline | null = null;
  deltaPipeline: GPUComputePipeline | null = null;
  trainingBindGroup: GPUBindGroup | null = null;

  // Buffers
  weightBuffer: GPUBuffer | null = null;
  inputBuffer: GPUBuffer | null = null;
  biasBuffer: GPUBuffer | null = null;
  outputBuffer: GPUBuffer | null = null;
  uniformBuffer: GPUBuffer | null = null;

  networkSize: number = 0;
  batchSize: number = 1;

  async init() {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No GPU adapter found');
    this.device = await adapter.requestDevice();

    const shaderModule = this.device.createShaderModule({ code: shaderCode });
    const trainingModule = this.device.createShaderModule({
      code: trainingShaderCode,
    });

    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });

    this.trainingPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: trainingModule, entryPoint: 'update_weights' },
    });

    this.deltaPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: trainingModule, entryPoint: 'calculate_deltas' },
    });

    console.log('GPUEngine initialized');
  }

  // Prepare buffers based on network size (N) and Batch Size (B)
  prepareBuffers(
    size: number,
    weights: Float32Array,
    biases: Float32Array,
    batchSize: number = 1
  ) {
    if (!this.device || !this.pipeline)
      throw new Error('GPUEngine not initialized');
    this.networkSize = size;
    this.batchSize = batchSize;

    // Create Buffers
    // Weights & Biases are shared (Size N or N*N)
    this.weightBuffer = this.createBuffer(
      weights,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.biasBuffer = this.createBuffer(
      biases,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    // Inputs & Outputs are Batched (Size N * B)
    const batchedSize = size * batchSize;
    this.inputBuffer = this.createBuffer(
      new Float32Array(batchedSize),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.outputBuffer = this.createBuffer(
      new Float32Array(batchedSize),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );

    // Dimensions Uniform: [Size, BatchSize]
    const dimArray = new Uint32Array([size, batchSize]);
    this.uniformBuffer = this.createBuffer(
      dimArray,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    // Bind Group
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weightBuffer } },
        { binding: 1, resource: { buffer: this.inputBuffer } },
        { binding: 2, resource: { buffer: this.biasBuffer } },
        { binding: 3, resource: { buffer: this.outputBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffer } },
      ],
    });
  }

  private createBuffer(
    data: Float32Array | Uint32Array,
    usage: number
  ): GPUBuffer {
    if (!this.device) throw new Error('Device null');
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage,
      mappedAtCreation: true,
    });
    if (data instanceof Float32Array) {
      new Float32Array(buffer.getMappedRange()).set(data);
    } else {
      new Uint32Array(buffer.getMappedRange()).set(data);
    }
    buffer.unmap();
    return buffer;
  }

  async runTick(inputs: Float32Array): Promise<Float32Array> {
    if (
      !this.device ||
      !this.pipeline ||
      !this.bindGroup ||
      !this.inputBuffer ||
      !this.outputBuffer
    ) {
      throw new Error('GPU buffers not ready');
    }

    if (inputs.length !== this.networkSize * this.batchSize) {
      throw new Error(
        `Input size mismatch. Expected ${
          this.networkSize * this.batchSize
        }, got ${inputs.length}`
      );
    }

    // Upload Input
    this.device.queue.writeBuffer(this.inputBuffer, 0, inputs as BufferSource);

    // Encode Command
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);

    // Dispatch (Size / WorkgroupSize, 1, BatchSize)
    const workgroupSize = 64;
    const workgroupCount = Math.ceil(this.networkSize / workgroupSize);
    passEncoder.dispatchWorkgroups(workgroupCount, 1, this.batchSize);
    passEncoder.end();

    // Read Output
    const size = inputs.byteLength;
    const gpuReadBuffer = this.device.createBuffer({
      size: size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(
      this.outputBuffer,
      0,
      gpuReadBuffer,
      0,
      size
    );

    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(gpuReadBuffer.getMappedRange());
    const output = new Float32Array(result); // Copy
    gpuReadBuffer.unmap();

    return output;
  }

  prepareTrainingBuffers(targets: Float32Array, learningRate: number) {
    if (
      !this.device ||
      !this.trainingPipeline ||
      !this.weightBuffer ||
      !this.outputBuffer ||
      !this.biasBuffer ||
      !this.uniformBuffer
    ) {
      throw new Error('GPU not ready for training');
    }

    if (targets.length !== this.networkSize * this.batchSize) {
      throw new Error(
        `Target size mismatch. Expected ${
          this.networkSize * this.batchSize
        }, got ${targets.length}`
      );
    }

    // Deltas & Targets are Batched (Size N * B)
    const batchedSize = this.networkSize * this.batchSize;
    this.deltaBuffer = this.createBuffer(
      new Float32Array(batchedSize),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    );
    this.targetBuffer = this.createBuffer(
      targets,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.paramBuffer = this.createBuffer(
      new Float32Array([learningRate]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    this.trainingBindGroup = this.device.createBindGroup({
      layout: this.trainingPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weightBuffer } },
        { binding: 1, resource: { buffer: this.outputBuffer } },
        { binding: 2, resource: { buffer: this.biasBuffer } },
        { binding: 3, resource: { buffer: this.deltaBuffer } },
        { binding: 4, resource: { buffer: this.targetBuffer } },
        { binding: 5, resource: { buffer: this.uniformBuffer } },
        { binding: 6, resource: { buffer: this.paramBuffer } },
      ],
    });
  }

  private subscribers: ((event: {
    type: 'loss' | 'epoch';
    value: number;
  }) => void)[] = [];

  subscribe(
    callback: (event: { type: 'loss' | 'epoch'; value: number }) => void
  ) {
    this.subscribers.push(callback);
    return () => {
      this.subscribers = this.subscribers.filter((s) => s !== callback);
    };
  }

  private emit(event: { type: 'loss' | 'epoch'; value: number }) {
    this.subscribers.forEach((cb) => cb(event));
  }

  async train(
    inputs: Float32Array,
    targets: Float32Array
  ): Promise<Float32Array> {
    // 1. Forward Pass
    const outputs = await this.runTick(inputs);

    // 2. Calculate Loss (MSE) on CPU for UI Feedback
    // Only feasible if batch size is small or we sample.
    // For demo, we just calc full MSE.
    let totalLoss = 0;
    for (let i = 0; i < outputs.length; i++) {
      // Only if target is valid? Assuming targets cover all neurons logic as per shader
      const t = targets[i];
      if (t > -998) {
        const diff = outputs[i] - t;
        totalLoss += 0.5 * diff * diff;
      }
    }
    const meanLoss = totalLoss / this.batchSize; // Approx
    this.emit({ type: 'loss', value: meanLoss });

    // 3. Backward Pass
    // Ensure buffers (deltas, targets) are ready?
    // Reuse prepareTrainingBuffers or assume already called?
    // Let's assume prepareTrainingBuffers was called ONCE before loop.
    // We just need to update TARGETS buffer!
    if (this.targetBuffer) {
      this.device?.queue.writeBuffer(
        this.targetBuffer,
        0,
        targets as BufferSource
      );
    }

    // Run Training Shaders
    await this.trainTick();

    this.emit({ type: 'epoch', value: 1 }); // Just tick count really
    return outputs;
  }

  async trainTick(deltas?: Float32Array): Promise<void> {
    if (
      !this.device ||
      !this.trainingPipeline ||
      !this.deltaPipeline ||
      !this.trainingBindGroup ||
      !this.deltaBuffer
    ) {
      throw new Error('Training not ready');
    }

    if (deltas && deltas.length > 0) {
      this.device.queue.writeBuffer(
        this.deltaBuffer,
        0,
        deltas as BufferSource
      );
    }

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    // Pass 1: Calculate Deltas (Batched)
    passEncoder.setPipeline(this.deltaPipeline);
    passEncoder.setBindGroup(0, this.trainingBindGroup);
    const workgroupSize = 64;
    const workgroupCount = Math.ceil(this.networkSize / workgroupSize);
    passEncoder.dispatchWorkgroups(workgroupCount, 1, this.batchSize);

    passEncoder.end();

    const updatePass = commandEncoder.beginComputePass();
    updatePass.setPipeline(this.trainingPipeline);
    updatePass.setBindGroup(0, this.trainingBindGroup); // Re-bind for new pass
    updatePass.dispatchWorkgroups(workgroupCount, 1, 1); // Not batched
    updatePass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  async injectInput(data: Float32Array): Promise<void> {
    if (!this.device || !this.inputBuffer) return;

    // We only write what we are given, usually just the first N inputs (Microphone bins)
    // If data is smaller than buffer, we use queue.writeBuffer which handles partial writes
    this.device.queue.writeBuffer(this.inputBuffer, 0, data as BufferSource);

    // Trigger a tick? Or let the outer loop do it?
    // Let's just update the buffer. The UI loop calls runTick() or similar.
  }
}
