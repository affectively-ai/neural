import { dash } from "@buley/dash";

export interface Neuron {
    id: string;
    type: 'input' | 'hidden' | 'output';
    bias: number;
    activation: string;
}

export interface Synapse {
    id: string;
    from_id: string;
    to_id: string;
    weight: number;
}

export class NeuronRepository {
    async create(neuron: Neuron): Promise<void> {
        await dash.execute(
            "INSERT INTO neurons (id, type, bias, activation) VALUES (?, ?, ?, ?)", 
            [neuron.id, neuron.type, neuron.bias, neuron.activation]
        );
    }
    
    // Feature: Add with semantic embedding
    async createWithSemantics(neuron: Neuron, description: string): Promise<void> {
        // We store the structured data normally
        await this.create(neuron);
        // And we map the ID to a semantic embedding in dash's hidden semantic store
        await dash.addWithEmbedding(neuron.id, description);
    }

    async getAll(): Promise<Neuron[]> {
        return await dash.execute("SELECT * FROM neurons") as Neuron[];
    }
}

export class SynapseRepository {
    async create(synapse: Synapse): Promise<void> {
        await dash.execute(
            "INSERT INTO synapses (id, from_id, to_id, weight) VALUES (?, ?, ?, ?)",
            [synapse.id, synapse.from_id, synapse.to_id, synapse.weight]
        );
    }

    async getAll(): Promise<Synapse[]> {
        return await dash.execute("SELECT * FROM synapses") as Synapse[];
    }
}
