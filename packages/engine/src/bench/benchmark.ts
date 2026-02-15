import { GPUEngine } from '../engine/gpu';
import { performance } from 'perf_hooks';

// Mock WebGPU types for Node environment if needed, likely handled by bun's specialized runtime or mocks if strictly node.
// However, since we are using Bun, we might need a headless WebGPU implementation or we run this in a real browser.
// REALITY CHECK: running WebGPU in a headless CI/Node environment usually requires 'headless-gl' or similar, but WebGPU is newer.
// Bun does not support WebGPU native out of the box yet.
// For this task, since the user is on Mac, we will assume they might run this via a browser test runner OR we simulate/mock for the "structure" of the benchmark
// if actual GPU access isn't available in the terminal.
// BUT: The roadmap implies real GPU benchmarks.
// Strategy: We will write the benchmark to be runnable. If it fails due to missing GPU in terminal,
// we'll note that it needs to be run in a browser context (e.g. via the web app or a test runner that supports it).
// actually, for the purpose of this agent, I'll implement it assuming the environment *might* support it or I'll add a check.

async function runBenchmark(
  label: string,
  networkSize: number,
  batchSize: number,
  iterations: number
) {
  console.log(`\n--- Benchmark: ${label} ---`);
  console.log(`Network: ${networkSize} Neurons, Batch: ${batchSize}`);

  const gpu = new GPUEngine();

  try {
    await gpu.init();
  } catch (e) {
    console.error('WebGPU Initialize Failed (Expected in non-browser env):', e);
    return;
  }

  // Prepare Data
  const weights = new Float32Array(networkSize * networkSize); // Full connectivity
  const biases = new Float32Array(networkSize);
  const inputs = new Float32Array(networkSize * batchSize);
  const targets = new Float32Array(networkSize * batchSize);

  // Init Buffers
  const startObj = performance.now();
  gpu.prepareBuffers(networkSize, weights, biases, batchSize);
  gpu.prepareTrainingBuffers(targets, 0.01);
  const initTime = performance.now() - startObj;
  console.log(`Initialization/Upload: ${initTime.toFixed(2)}ms`);

  // Warmup
  await gpu.runTick(inputs);

  // Measure Inference
  const startInf = performance.now();
  for (let i = 0; i < iterations; i++) {
    await gpu.runTick(inputs);
  }
  const endInf = performance.now();
  const infTime = endInf - startInf;
  const infOPS = (iterations * batchSize) / (infTime / 1000);
  console.log(`Inference: ${infTime.toFixed(2)}ms for ${iterations} ticks`);
  console.log(`Throughput: ${infOPS.toFixed(0)} samples/sec`);

  // Measure Training
  const startTrain = performance.now();
  for (let i = 0; i < iterations; i++) {
    await gpu.trainTick();
  }
  const endTrain = performance.now();
  const trainTime = endTrain - startTrain;
  const trainOPS = (iterations * batchSize) / (trainTime / 1000);
  console.log(`Training: ${trainTime.toFixed(2)}ms for ${iterations} ticks`);
  console.log(`Throughput: ${trainOPS.toFixed(0)} samples/sec`);
}

async function main() {
  // Small
  await runBenchmark('Small', 100, 1, 100);

  // Medium
  await runBenchmark('Medium (Batched)', 1000, 32, 50);

  // Large
  await runBenchmark('Large (Batched)', 5000, 64, 20);
}

// Check for WebGPU polyfill or mock if running in Node without headers
if (!global.navigator?.gpu) {
  console.log(
    'No WebGPU detected in global scope. Mocking for CLI structure verification...'
  );
  // @ts-ignore
  global.navigator = {
    gpu: {
      ...({} as any as GPU),
      requestAdapter: async () => ({
        ...({} as any as GPUAdapter), // Force cast for mock
        requestDevice: async () => ({
          ...({} as any as GPUDevice),
          createShaderModule: () => ({} as unknown as GPUShaderModule),
          createComputePipeline: () =>
            ({
              getBindGroupLayout: () => ({} as unknown as GPUBindGroupLayout),
            } as unknown as GPUComputePipeline),
          createBuffer: (d: any) =>
            ({
              getMappedRange: () => new ArrayBuffer(d.size),
              unmap: () => {},
              mapAsync: async () => {},
            } as unknown as GPUBuffer),
          createBindGroup: () => ({} as unknown as GPUBindGroup),
          createCommandEncoder: () =>
            ({
              beginComputePass: () =>
                ({
                  setPipeline: () => {},
                  setBindGroup: () => {},
                  dispatchWorkgroups: () => {},
                  end: () => {},
                } as unknown as GPUComputePassEncoder),
              copyBufferToBuffer: () => {},
              finish: () => ({} as any as GPUCommandBuffer),
            } as unknown as GPUCommandEncoder),
          queue: {
            writeBuffer: () => {},
            submit: () => {},
          } as unknown as GPUQueue,
        }),
      }),
    },
  };
  // @ts-ignore
  global.GPUBufferUsage = {
    STORAGE: 1,
    COPY_DST: 2,
    COPY_SRC: 4,
    UNIFORM: 8,
    MAP_READ: 16,
  };
  // @ts-ignore
  global.GPUMapMode = { READ: 1 };
}

main();
