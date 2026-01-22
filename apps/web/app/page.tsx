'use client';

import { useEffect, useState } from 'react';
// We import dynamically or use client-side logic because Engine relies on `navigator.gpu` and `window`
// However, the Engine package might be isomorphic or need dynamic import if it has side effects.
// For now, assuming basic import works but initialization must be in useEffect.

export default function Home() {
  const [logs, setLogs] = useState<string[]>([]);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    // Dynamic import to avoid SSR issues with WebGPU/WASM
    import('@neural/engine').then(async (engine) => {
      addLog("Initializing Neural Engine...");
      
      try {
        // We'll expose the runTick or init method from the engine
        // Assuming we need to add an export to engine/src/index.ts that creates the engine instance
        await engine.init(); 
        addLog("Engine Ready. WebGPU + Dash Connected.");
        setIsReady(true);
      } catch (e: any) {
        addLog(`Error: ${e.message}`);
      }
    });
  }, []);

  const addLog = (msg: string) => setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-zinc-950 text-emerald-400 font-mono">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold mb-8">Neural 2.0: The Transparent Brain</h1>
      </div>

      <div className="relative flex place-items-center">
        <div className="w-[600px] h-[400px] border border-emerald-900 bg-black/50 rounded-lg p-4 font-mono text-sm overflow-y-auto font-bold shadow-[0_0_50px_-12px_rgba(16,185,129,0.25)]">
          {logs.map((log, i) => (
            <div key={i} className="mb-1">{log}</div>
          ))}
          {!isReady && <div className="animate-pulse">_</div>}
        </div>
      </div>

      <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-3 lg:text-left gap-4">
        <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-emerald-700 hover:bg-emerald-900/20">
          <h2 className="mb-3 text-2xl font-semibold">WebGPU Compute</h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Running custom WGSL shaders for massive parallelism.
          </p>
        </div>
        <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-emerald-700 hover:bg-emerald-900/20">
          <h2 className="mb-3 text-2xl font-semibold">Dash Persistence</h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Local-first SQLite + OPFS storage with vector search.
          </p>
        </div>
        <div className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-emerald-700 hover:bg-emerald-900/20">
          <h2 className="mb-3 text-2xl font-semibold">Semantic Graph</h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Neurons tagged with natural language embeddings.
          </p>
        </div>
      </div>
    </main>
  );
}
