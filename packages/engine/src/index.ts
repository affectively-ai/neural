import { dash } from "@buley/dash";
import { initializeSchema } from "./db/schema";
import { NeuronRepository, SynapseRepository } from "./db/repository";
import { GPUEngine } from "./engine/gpu";
import { Translator } from "./engine/translator";

export class NeuralEngine {
    gpu: GPUEngine;
    neuronRepo: NeuronRepository;
    synapseRepo: SynapseRepository;
    private translator: Translator;

    constructor() {
        this.gpu = new GPUEngine();
        this.neuronRepo = new NeuronRepository();
        this.synapseRepo = new SynapseRepository();
        this.translator = new Translator();
    }

    // Cache
    private neurons: any[] = [];
    private synapses: any[] = [];

    async init() {
        console.log("Neural 2.0 Engine Initializing...");
        
        // 1. Persistence
        await dash.ready();
        await initializeSchema();
        
        // 2. Compute
        await this.gpu.init();
        this.gpu.batchSize = 2; // Default to mini-batch of 2 for demo

        // 3. Hydration
        this.neurons = await this.neuronRepo.getAll();
        this.synapses = await this.synapseRepo.getAll();

        if (this.neurons.length === 0) {
            console.log("Seeding test network...");
            const n1 = "n1-" + crypto.randomUUID();
            const n2 = "n2-" + crypto.randomUUID();
            await this.neuronRepo.create({ id: n1, type: 'input', bias: 0, activation: 'tanh' });
            await this.neuronRepo.create({ id: n2, type: 'output', bias: 0.5, activation: 'tanh' });
            await this.synapseRepo.create({ id: crypto.randomUUID(), from_id: n1, to_id: n2, weight: 0.8 });
            
            // Seed more for visualizer? 
            // Let's create a random cluster for demo purposes if empty
            for(let i=0; i<50; i++) {
                 await this.neuronRepo.create({ id: `auto-${i}`, type: 'hidden', bias: 0, activation: 'tanh' });
            }
            // Connect them
            const all = await this.neuronRepo.getAll();
            for(let i=0; i<50; i++) {
                const s = all[Math.floor(Math.random() * all.length)].id;
                const t = all[Math.floor(Math.random() * all.length)].id;
                if(s!==t) await this.synapseRepo.create({ id: crypto.randomUUID(), from_id: s, to_id: t, weight: Math.random() });
            }

            this.neurons = await this.neuronRepo.getAll();
            this.synapses = await this.synapseRepo.getAll();
        }

        // 4. Compile to GPU
        console.log(`Compiling graph: ${this.neurons.length} neurons, ${this.synapses.length} synapses`);
        const data = this.translator.flatten(this.neurons, this.synapses);

        this.gpu.prepareBuffers(data.size, data.weights, data.biases, this.gpu.batchSize);
        // Also prepare training buffers!
        // Init target buffer with zeros
        this.gpu.prepareTrainingBuffers(new Float32Array(data.size * this.gpu.batchSize), 0.1);

        console.log("Engine Ready.");
        return data;
    }

    getGraphData() {
        // Map ID -> Index
        const map = new Map<string, number>();
        this.neurons.forEach((n, i) => map.set(n.id, i));

        const edges = this.synapses.map(s => ({
            id: s.id,
            source: map.get(s.from_id) || 0,
            target: map.get(s.to_id) || 0,
            weight: s.weight
        }));

        return {
            nodeCount: this.neurons.length,
            edges
        };
    }

    async deleteSynapse(id: string) {
        console.log(`Lesioning synapse: ${id}`);
        await this.synapseRepo.delete(id);
        
        // Update Cache
        this.synapses = this.synapses.filter(s => s.id !== id);
        
        // Recompile (Heavy!)
        // In a real app we'd just zero the weight in buffer
        // But for "The Visible Brain" seeing it disappear is cooler.
        const data = this.translator.flatten(this.neurons, this.synapses);
        this.gpu.prepareBuffers(data.size, data.weights, data.biases, this.gpu.batchSize);
        // Reset training buffers too to be safe/simple
        this.gpu.prepareTrainingBuffers(new Float32Array(data.size * this.gpu.batchSize), 0.1);

        return this.getGraphData();
    }

    exportGraph() {
        return {
            version: "2.0",
            neurons: this.neurons,
            synapses: this.synapses
        };
    }

    async importGraph(data: any) {
        if (!data.neurons || !data.synapses) throw new Error("Invalid graph data");
        
        console.log("Importing graph...");
        
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
        
        console.log(`Compiling imported graph: ${this.neurons.length} neurons, ${this.synapses.length} synapses`);
        const graph = this.translator.flatten(this.neurons, this.synapses);
        this.gpu.prepareBuffers(graph.size, graph.weights, graph.biases, this.gpu.batchSize);
        this.gpu.prepareTrainingBuffers(new Float32Array(graph.size * this.gpu.batchSize), 0.1);
        


    async injectInput(data: Float32Array) {
        // Map data to input neurons
        // In simulation, we assume first N neurons are inputs.
        // Or we just overwrite the first N values of the input buffer.
        await this.gpu.injectInput(data);
    }
}

// Keep a standalone init for backward compatibility or simple scripts if needed
export async function init() {
    const engine = new NeuralEngine();
    return engine.init();
}
