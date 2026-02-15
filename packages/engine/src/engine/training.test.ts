import { expect, test, describe, mock } from 'bun:test';
import { GPUEngine } from './gpu';

// Mock GPU globals again (copy from gpu.test.ts or import shared mock if we had one)
// Ideally refactor mocks to a setup file. For now, inline.
const mockDevice = {
  createShaderModule: mock(() => ({})),
  createComputePipeline: mock(() => ({
    getBindGroupLayout: mock(() => ({})),
  })),
  createBuffer: mock((desc: any) => ({
    getMappedRange: () => new ArrayBuffer(desc.size),
    unmap: () => {},
    mapAsync: async () => {},
  })),
  createBindGroup: mock(() => ({})),
  createCommandEncoder: mock(() => ({
    beginComputePass: mock(() => ({
      setPipeline: mock(() => {}),
      setBindGroup: mock(() => {}),
      dispatchWorkgroups: mock(() => {}),
      end: mock(() => {}),
    })),
    copyBufferToBuffer: mock(() => {}),
    finish: mock(() => ({})),
  })),
  queue: {
    writeBuffer: mock(() => {}),
    submit: mock(() => {}),
  },
};

const mockAdapter = {
  requestDevice: mock(async () => mockDevice),
};

// @ts-ignore
Object.defineProperty(global, 'navigator', {
  value: { gpu: { requestAdapter: mock(async () => mockAdapter) } },
  writable: true,
  configurable: true,
});
// @ts-ignore
Object.defineProperty(global, 'GPUBufferUsage', {
  value: {
    STORAGE: 1,
    COPY_DST: 2,
    COPY_SRC: 4,
    UNIFORM: 8,
    MAP_READ: 16,
  },
  writable: true,
  configurable: true,
});
// @ts-ignore
Object.defineProperty(global, 'GPUMapMode', {
  value: { READ: 1 },
  writable: true,
  configurable: true,
});

describe('Training Loop', () => {
  test('prepareTrainingBuffers allocation (Batch=1)', async () => {
    const gpu = new GPUEngine();
    await gpu.init();

    gpu.networkSize = 10;
    gpu.batchSize = 1;
    gpu.weightBuffer = {} as any;
    gpu.outputBuffer = {} as any;
    gpu.biasBuffer = {} as any;
    gpu.uniformBuffer = {} as any;

    gpu.prepareTrainingBuffers(new Float32Array(10), 0.01);

    expect(mockDevice.createBuffer).toHaveBeenCalled();
    expect(gpu.deltaBuffer).toBeDefined();
    expect(gpu.targetBuffer).toBeDefined();
  });

  test('trainTick dispatch (Batch=1)', async () => {
    const gpu = new GPUEngine();
    await gpu.init();
    gpu.networkSize = 10;
    gpu.batchSize = 1;
    gpu.weightBuffer = {} as any;
    gpu.outputBuffer = {} as any;
    gpu.biasBuffer = {} as any;
    gpu.uniformBuffer = {} as any;

    gpu.prepareTrainingBuffers(new Float32Array(10), 0.01);

    await gpu.trainTick(); // Uses internal buffer/logic

    // Should dispatch twice (Delta Calc + Weight Update)
    // We can check calls to createCommandEncoder -> beginComputePass
    // const encoder = mockDevice.createCommandEncoder.mock.results.at(-1)?.value;
    // Check for pass calls if needed, otherwise ignore for now
    // const pass = encoder?.beginComputePass.mock.results.at(-1)?.value;
    // mock logic is a bit simple, let's just check overall calls
    expect(mockDevice.queue.submit).toHaveBeenCalled();
  });

  test('prepareTrainingBuffers allocation (Batch=2)', async () => {
    const gpu = new GPUEngine();
    await gpu.init();

    gpu.networkSize = 10;
    gpu.batchSize = 2; // Test Batch > 1
    gpu.weightBuffer = {} as any;
    gpu.outputBuffer = {} as any;
    gpu.biasBuffer = {} as any;
    gpu.uniformBuffer = {} as any;

    // Target size must be 20
    gpu.prepareTrainingBuffers(new Float32Array(20), 0.01);

    expect(mockDevice.createBuffer).toHaveBeenCalled();
    expect(gpu.deltaBuffer).toBeDefined();
    // Check delta buffer size? Mock doesn't store it easily accessible, but verify it didn't throw.
  });
});
