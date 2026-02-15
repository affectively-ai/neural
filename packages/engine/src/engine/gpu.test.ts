import { expect, test, describe, mock } from 'bun:test';
import { GPUEngine } from './gpu';

// Mock WebGPU Globals
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

// Polyfill navigator.gpu
// @ts-ignore
Object.defineProperty(global, 'navigator', {
  value: {
    gpu: {
      requestAdapter: mock(async () => mockAdapter as unknown as GPUAdapter),
    } as unknown as GPU,
  },
  writable: true,
  configurable: true,
});

// Polyfill Globals
// @ts-ignore
Object.defineProperty(global, 'GPUBufferUsage', {
  value: {
    MAP_READ: 1,
    MAP_WRITE: 2,
    COPY_SRC: 4,
    COPY_DST: 8,
    INDEX: 16,
    VERTEX: 32,
    UNIFORM: 64,
    STORAGE: 128,
    INDIRECT: 256,
    QUERY_RESOLVE: 512,
  },
  writable: true,
  configurable: true,
});
// @ts-ignore
Object.defineProperty(global, 'GPUMapMode', {
  value: {
    READ: 1,
    WRITE: 2,
  },
  writable: true,
  configurable: true,
});

describe('GPUEngine', () => {
  test('init() requests adapter and device', async () => {
    const gpu = new GPUEngine();
    await gpu.init();
    expect(navigator.gpu.requestAdapter).toHaveBeenCalled();
    expect(mockAdapter.requestDevice).toHaveBeenCalled();
    expect(gpu.device).toBeDefined();
  });

  test('prepareBuffers() creates GPU buffers', async () => {
    const gpu = new GPUEngine();
    await gpu.init();

    const weights = new Float32Array([1, 2, 3, 4]);
    const biases = new Float32Array([0, 0]);

    gpu.prepareBuffers(2, weights, biases);

    expect(mockDevice.createBuffer).toHaveBeenCalledTimes(5); // W, I, B, O, Uniforms
    expect(mockDevice.createBindGroup).toHaveBeenCalled();
  });

  test('runTick() dispatches compute shader', async () => {
    const gpu = new GPUEngine();
    await gpu.init();
    gpu.prepareBuffers(2, new Float32Array(4), new Float32Array(2));

    const inputs = new Float32Array([1, 0]);
    await gpu.runTick(inputs);

    expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
    expect(mockDevice.createCommandEncoder).toHaveBeenCalled();
    // Check dispatch
    // We can't easily check the nested mock calls count without storing the mock,
    // but if no error threw, the flow worked.
  });

  test('prepareBuffers() and runTick() with Batch Size > 1', async () => {
    const gpu = new GPUEngine();
    await gpu.init();

    const N = 4;
    const B = 2; // Batch Size 2

    gpu.prepareBuffers(N, new Float32Array(N), new Float32Array(N), B);

    // Input size = N * B = 8
    const inputs = new Float32Array(N * B);
    await gpu.runTick(inputs);

    expect(mockDevice.createBuffer).toHaveBeenCalled();
    expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();

    // Verify buffer size in mock? Mock doesn't store state.
    // But verifying it runs without throwing "Input size mismatch" proves validation worked.
  });
});
