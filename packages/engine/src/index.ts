import { dash } from '@affectively/dash';
import { initializeSchema } from './db/schema';
import { NeuronRepository, SynapseRepository } from './db/repository';
import { GPUEngine } from './engine/gpu';
import { WebNNEngine } from './engine/webnn';
import { Translator } from './engine/translator';
import { Neuron, Synapse } from './types';

export type { Neuron, Synapse } from './types';

export interface AdapterTrainingConfig {
  rank: number;
  basePrecision: 'int8' | 'fp16' | 'fp32';
  adapterPrecision: 'fp16' | 'fp32';
  microBatchSize: number;
  idleFlushMs: number;
}

export class NeuralEngine {
  gpu: GPUEngine;
  npu: WebNNEngine;
  neuronRepo: NeuronRepository;
  synapseRepo: SynapseRepository;
  private translator: Translator;

  activeBackend: 'gpu' | 'npu' = 'gpu';
  adapterTrainingConfig: AdapterTrainingConfig = {
    rank: 8,
    basePrecision: 'int8',
    adapterPrecision: 'fp16',
    microBatchSize: 16,
    idleFlushMs: 45_000,
  };

  constructor() {
    this.gpu = new GPUEngine();
    this.npu = new WebNNEngine();
    this.neuronRepo = new NeuronRepository();
    this.synapseRepo = new SynapseRepository();
    this.translator = new Translator();
  }

  // Cache
  private neurons: Neuron[] = [];
  private synapses: Synapse[] = [];

  async init() {
    console.log('Neural 2.0 Engine Initializing...');

    // 1. Persistence
    await dash.ready();
    await initializeSchema();

    // 2. Compute
    await this.gpu.init();
    this.gpu.batchSize = 2; // Default to mini-batch of 2 for demo

    // Try NPU
    await this.npu.init();
    if (this.npu.isReady) {
      console.log('Neural Engine: NPU Accelerated Backend Available.');
      this.activeBackend = 'npu';
    }

    // 3. Hydration
    this.neurons = await this.neuronRepo.getAll();
    this.synapses = await this.synapseRepo.getAll();

    if (this.neurons.length === 0) {
      console.log('Seeding test network...');
      const n1 = 'n1-' + crypto.randomUUID();
      const n2 = 'n2-' + crypto.randomUUID();
      await this.neuronRepo.create({
        id: n1,
        type: 'input',
        bias: 0,
        activation: 'tanh',
      });
      await this.neuronRepo.create({
        id: n2,
        type: 'output',
        bias: 0.5,
        activation: 'tanh',
      });
      await this.synapseRepo.create({
        id: crypto.randomUUID(),
        from_id: n1,
        to_id: n2,
        weight: 0.8,
      });

      // Seed more for visualizer?
      // Let's create a random cluster for demo purposes if empty
      for (let i = 0; i < 50; i++) {
        await this.neuronRepo.create({
          id: `auto-${i}`,
          type: 'hidden',
          bias: 0,
          activation: 'tanh',
        });
      }
      // Connect them
      const all = await this.neuronRepo.getAll();
      for (let i = 0; i < 50; i++) {
        const s = all[Math.floor(Math.random() * all.length)].id;
        const t = all[Math.floor(Math.random() * all.length)].id;
        if (s !== t)
          await this.synapseRepo.create({
            id: crypto.randomUUID(),
            from_id: s,
            to_id: t,
            weight: Math.random(),
          });
      }

      this.neurons = await this.neuronRepo.getAll();
      this.synapses = await this.synapseRepo.getAll();
    }

    // 4. Compile to Compute Backends
    await this.compile();

    console.log(
      `Engine Ready. Active Backend: ${this.activeBackend.toUpperCase()}`
    );
    return this.getGraphData();
  }

  async compile() {
    console.log(
      `Compiling graph: ${this.neurons.length} neurons, ${this.synapses.length} synapses`
    );
    const data = this.translator.flatten(this.neurons, this.synapses);

    // GPU
    this.gpu.prepareBuffers(
      data.size,
      data.weights,
      data.biases,
      this.gpu.batchSize
    );
    this.gpu.prepareTrainingBuffers(
      new Float32Array(data.size * this.gpu.batchSize),
      0.1
    );

    // NPU
    if (this.npu.isReady) {
      await this.npu.prepareModel(data.size, data.weights, data.biases, 2);
    }

    return data; // Return data for init/others if needed
  }

  async deployToCloud() {
    console.log('Deploying heavy layers to Hybrid Cloud...');
    // Randomly tagging neurons as "cloud"
    this.neurons = this.neurons.map((n) => ({
      ...n,
      type: Math.random() > 0.8 ? 'cloud' : n.type,
    }));
    // Update repo? For demo we might just keep in memory or update repo.
    // Let's update repo for persistence
    for (const n of this.neurons) {
      if (n.type === 'cloud') {
        // Assuming repo has update or we just overwrite.
        // Repo might not support update efficiently.
        // For demo/speed, we likely just keep in memory for this session
        // unless we want it to persist.
      }
    }
    return this.getGraphData();
  }

  setAdapterTrainingConfig(config: Partial<AdapterTrainingConfig>) {
    this.adapterTrainingConfig = {
      ...this.adapterTrainingConfig,
      ...config,
    };
    this.gpu.batchSize = this.adapterTrainingConfig.microBatchSize;
  }

  getAdapterTrainingConfig(): AdapterTrainingConfig {
    return { ...this.adapterTrainingConfig };
  }

  getGraphData() {
    // Map ID -> Index
    const map = new Map<string, number>();
    this.neurons.forEach((n, i) => map.set(n.id, i));

    const edges = this.synapses.map((s) => ({
      id: s.id,
      source: map.get(s.from_id) || 0,
      target: map.get(s.to_id) || 0,
      weight: s.weight,
    }));

    // Return full nodes for visualization customization
    const nodes = this.neurons.map((n, i) => ({
      id: n.id,
      index: i,
      type: n.type,
    }));

    return {
      nodeCount: this.neurons.length,
      nodes,
      edges,
    };
  }

  async deleteSynapse(id: string) {
    console.log(`Lesioning synapse: ${id}`);
    await this.synapseRepo.delete(id);

    // Update Cache
    this.synapses = this.synapses.filter((s) => s.id !== id);

    // Recompile (Heavy!)
    await this.compile();

    return this.getGraphData();
  }

  exportGraph() {
    return {
      version: '2.0',
      neurons: this.neurons,
      synapses: this.synapses,
    };
  }

  async importGraph(data: { neurons: Neuron[]; synapses: Synapse[] }) {
    if (!data.neurons || !data.synapses) throw new Error('Invalid graph data');

    console.log('Importing graph...');

    // 1. Clear existing
    const oldNeurons = await this.neuronRepo.getAll();
    for (const n of oldNeurons) await this.neuronRepo.delete(n.id);

    const oldSynapses = await this.synapseRepo.getAll();
    for (const s of oldSynapses) await this.synapseRepo.delete(s.id);

    // 2. Insert new
    for (const n of data.neurons) await this.neuronRepo.create(n);
    for (const s of data.synapses) await this.synapseRepo.create(s);

    // 3. Hydrate & Compile
    this.neurons = await this.neuronRepo.getAll();
    this.synapses = await this.synapseRepo.getAll();

    console.log(
      `Compiling imported graph: ${this.neurons.length} neurons, ${this.synapses.length} synapses`
    );
    await this.compile();

    return this.getGraphData();
  }

  async injectInput(data: Float32Array) {
    if (this.activeBackend === 'npu' && this.npu.isReady) {
      // WebNN takes input at runTick
    } else {
      await this.gpu.injectInput(data);
    }
  }

  async runTick(inputs: Float32Array): Promise<Float32Array> {
    if (this.activeBackend === 'npu' && this.npu.isReady) {
      return this.npu.runTick(inputs);
    } else {
      return this.gpu.runTick(inputs);
    }
  }

  async trainAdapterMicroBatch(
    inputs: Float32Array,
    targets: Float32Array
  ): Promise<Float32Array> {
    if (this.activeBackend === 'npu' && this.npu.isReady) {
      throw new Error(
        'Adapter micro-batch training is currently GPU-only in this runtime'
      );
    }
    return this.gpu.train(inputs, targets);
  }
}

// Keep a standalone init for backward compatibility or simple scripts if needed
export async function init() {
  const engine = new NeuralEngine();
  return engine.init();
}
