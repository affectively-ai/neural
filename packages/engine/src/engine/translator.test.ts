import { expect, test, describe } from 'bun:test';
import { Translator } from './translator';
import { Neuron, Synapse } from '../types';
import { mock } from 'bun:test';

mock.module('@affectively/dash', () => ({
  dash: {},
}));

describe('Translator', () => {
  test('flatten() correctly converts graph to matrices', () => {
    const translator = new Translator();

    const neurons: Neuron[] = [
      { id: 'n1', type: 'input', bias: 0.5, activation: 'tanh' },
      { id: 'n2', type: 'output', bias: -0.2, activation: 'tanh' },
    ];

    const synapses: Synapse[] = [
      // Connection n1 -> n2 (weight 0.8)
      { id: 's1', from_id: 'n1', to_id: 'n2', weight: 0.8 },
    ];

    const result = translator.flatten(neurons, synapses);

    expect(result.size).toBe(2);

    // Check Biases
    expect(result.biases[0]).toBe(0.5);
    expect(result.biases[1]).toBeCloseTo(-0.2);

    // Check Weights Matrix (Size 2x2 = 4 elements)
    // Matrix is flattened: row * size + col
    // n1 is index 0, n2 is index 1
    // s1 is 0->1.
    // If weights[to * size + from], then weights[1 * 2 + 0] = weights[2] should be 0.8

    // Index 2 is row 1, col 0. (Target n2, Source n1)
    expect(result.weights[2]).toBeCloseTo(0.8);

    // Others should be 0
    expect(result.weights[0]).toBe(0); // 0->0
    expect(result.weights[1]).toBe(0); // 0->1 (n2 -> n1) - Wait, if 1*2+0=2, then 0*2+1=1.
    // Let's re-verify my translator logic.
    // fromIdx = 0, toIdx = 1.
    // flatIndex = (toIdx * size) + fromIdx = (1 * 2) + 0 = 2.

    expect(result.weights[1]).toBe(0);
    expect(result.weights[3]).toBe(0); // 1->1
  });

  test('handles empty graph', () => {
    const translator = new Translator();
    const result = translator.flatten([], []);
    expect(result.size).toBe(0);
    expect(result.weights.length).toBe(0);
  });
});
