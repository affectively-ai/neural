import { useState } from 'react';
import type { Neuron, Synapse } from '@buley/neural';

interface NeuralGraphData {
  version: string;
  neurons: Neuron[];
  synapses: Synapse[];
}

// Preset Architectures
const PRESETS: { name: string; desc: string; data: NeuralGraphData }[] = [
  {
    name: 'XOR Logic Gate',
    desc: 'Classic non-linear problem. 2 Inputs, 1 Hidden Layer (2 units), 1 Output.',
    data: {
      version: '2.0',
      neurons: [
        { id: 'in1', type: 'input', bias: 0, activation: 'tanh' },
        { id: 'in2', type: 'input', bias: 0, activation: 'tanh' },
        { id: 'h1', type: 'hidden', bias: -0.5, activation: 'tanh' },
        { id: 'h2', type: 'hidden', bias: 0.5, activation: 'tanh' },
        { id: 'out', type: 'output', bias: 0, activation: 'tanh' },
      ],
      synapses: [
        { id: 's1', from_id: 'in1', to_id: 'h1', weight: 1 },
        { id: 's2', from_id: 'in1', to_id: 'h2', weight: 1 },
        { id: 's3', from_id: 'in2', to_id: 'h1', weight: 1 },
        { id: 's4', from_id: 'in2', to_id: 'h2', weight: 1 },
        { id: 's5', from_id: 'h1', to_id: 'out', weight: 1 },
        { id: 's6', from_id: 'h2', to_id: 'out', weight: -1 },
      ],
    },
  },
  {
    name: 'Recurrent Loop',
    desc: 'A simple memory circuit where a neuron feeds back into itself.',
    data: {
      version: '2.0',
      neurons: [
        { id: 'pulse', type: 'input', bias: 0, activation: 'tanh' },
        { id: 'memory', type: 'hidden', bias: 0, activation: 'tanh' },
        { id: 'read', type: 'output', bias: 0, activation: 'tanh' },
      ],
      synapses: [
        { id: 'feed', from_id: 'pulse', to_id: 'memory', weight: 1 },
        { id: 'loop', from_id: 'memory', to_id: 'memory', weight: 0.9 }, // Decay
        { id: 'out', from_id: 'memory', to_id: 'read', weight: 1 },
      ],
    },
  },
];

export function ModelZoo({
  onLoad,
}: {
  onLoad: (data: NeuralGraphData) => void;
}) {
  return (
    <div className="glass-panel p-6 rounded-xl border-brand-primary/20 bg-brand-primary/5">
      <h3 className="text-lg font-semibold text-brand-primary mb-4">
        Model Zoo
      </h3>
      <div className="space-y-3">
        {PRESETS.map((p, i) => (
          <div
            key={i}
            onClick={() => onLoad(p.data)}
            className="p-3 rounded-lg border border-white/10 hover:border-brand-primary/50 bg-black/40 cursor-pointer transition-all hover:translate-x-1 group"
          >
            <div className="flex justify-between items-center mb-1">
              <span className="font-mono text-xs font-bold text-white group-hover:text-brand-primary">
                {p.name}
              </span>
              <span className="text-[10px] text-brand-secondary bg-brand-secondary/10 px-1 rounded">
                PRESET
              </span>
            </div>
            <p className="text-[10px] text-neutral-400 leading-tight">
              {p.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ControlPanel({
  onExport,
  onImport,
}: {
  onExport: () => void;
  onImport: (data: NeuralGraphData) => void;
}) {
  return (
    <div className="flex gap-2">
      <button
        onClick={onExport}
        className="glass-button px-3 py-1 rounded text-xs text-white hover:bg-white/10 flex items-center gap-2"
      >
        <svg
          className="w-3 h-3"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
          ></path>
        </svg>
        Export JSON
      </button>
      <label className="glass-button px-3 py-1 rounded text-xs text-white hover:bg-white/10 flex items-center gap-2 cursor-pointer">
        <svg
          className="w-3 h-3"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
          ></path>
        </svg>
        Import JSON
        <input
          type="file"
          className="hidden"
          accept=".json"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
              try {
                const data = JSON.parse(
                  ev.target?.result as string
                ) as NeuralGraphData;
                onImport(data);
              } catch (err) {
                alert('Invalid JSON');
              }
            };
            reader.readAsText(file);
          }}
        />
      </label>
    </div>
  );
}
