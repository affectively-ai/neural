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
