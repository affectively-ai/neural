import { dash } from "@buley/dash";
import { initializeSchema } from "./db/schema";
import { NeuronRepository, SynapseRepository } from "./db/repository";
import { GPUEngine } from "./engine/gpu";
import { Translator } from "./engine/translator";

export async function init() {
    console.log("Neural 2.0 Engine Initializing...");
    
    // 1. Persistence
    await dash.ready();
    await initializeSchema();
    
    const neuronRepo = new NeuronRepository();
    const synapseRepo = new SynapseRepository();

    // 2. Compute
    const gpu = new GPUEngine();
    await gpu.init();

    // 3. Hydration
    let neurons = await neuronRepo.getAll();
    let synapses = await synapseRepo.getAll();

    if (neurons.length === 0) {
        console.log("Seeding test network...");
        const n1 = "n1-" + crypto.randomUUID();
        const n2 = "n2-" + crypto.randomUUID();
        await neuronRepo.create({ id: n1, type: 'input', bias: 0, activation: 'tanh' });
        await neuronRepo.create({ id: n2, type: 'output', bias: 0.5, activation: 'tanh' });
        await synapseRepo.create({ id: crypto.randomUUID(), from_id: n1, to_id: n2, weight: 0.8 });
        
        neurons = await neuronRepo.getAll();
        synapses = await synapseRepo.getAll();
    }

    // 4. Compile to GPU
    console.log(`Compiling graph: ${neurons.length} neurons, ${synapses.length} synapses`);
    const translator = new Translator();
    const data = translator.flatten(neurons, synapses);

    gpu.prepareBuffers(data.size, data.weights, data.biases);

    // 5. Run Live Loop (Simulated)
    console.log("Starting Inference Tick...");
    // Let's set input neuron (index 0) to 1.0
    data.initialValues[0] = 1.0; 
    
    const result = await gpu.runTick(data.initialValues);
    console.log("Inference Result (GPU Output):", result);
}
