'use client';

import { useEffect, useState, useRef } from 'react';
import dynamic from 'next/dynamic';

const Visualizer = dynamic(() => import('../components/Visualizer'), { 
    ssr: false, 
    loading: () => <div className="text-white/50 animate-pulse p-8">Initializing Neural Renderer...</div> 
});

import { ModelZoo, ControlPanel } from '../components/ModelZoo';

export default function Home() {
  const [logs, setLogs] = useState<string[]>([]);
  const [isReady, setIsReady] = useState(false);
  const [loss, setLoss] = useState<number | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [graphData, setGraphData] = useState<any>(null);
  
  const engineRef = useRef<any>(null);
  const trainingRef = useRef(false);

  // Audio State
  const [isListening, setIsListening] = useState(false);
  const audioRef = useRef<AudioInput | null>(null);

  const toggleListening = async () => {
      // Init audio if needed
      if (!audioRef.current && (typeof window !== 'undefined')) {
          const { AudioInput } = await import('../lib/audio');
          audioRef.current = new AudioInput();
      }
      
      if (isListening) {
          audioRef.current?.stop();
          setIsListening(false);
          addLog("Microphone Disconnected.");
      } else {
          try {
              if (audioRef.current) {
                await audioRef.current.start();
                setIsListening(true);
                addLog("Microphone Connected. Listening...");
                requestAnimationFrame(audioLoop);
              }
          } catch (e) {
              addLog("Mic Error: Permission Denied?");
          }
      }
  };

  const audioLoop = () => {
      if (!audioRef.current || !engineRef.current) return;
      
      // We check the ref directly for the loop condition to avoid stale closures
      if (!audioRef.current.isListening) return;
      
      const freq = audioRef.current.getFrequencyData();
      if (freq.length > 0) {
          engineRef.current.injectInput(freq);
          
          if (!trainingRef.current) {
              const size = engineRef.current.gpu.networkSize * engineRef.current.gpu.batchSize;
              const fullInput = new Float32Array(size);
              fullInput.set(freq); 
              engineRef.current.gpu.runTick(fullInput);
          }
      }
      
      requestAnimationFrame(audioLoop);
  };
  const handleExport = () => {
      if (!engineRef.current) return;
      const data = engineRef.current.exportGraph();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `neural-network-${new Date().toISOString()}.json`;
      a.click();
      addLog("Model exported to JSON.");
  };

  const handleImport = async (data: any) => {
      if (!engineRef.current) return;
      try {
          addLog("Loading external model...");
          const newData = await engineRef.current.importGraph(data);
          setGraphData(newData);
          addLog("Model imported successfully.");
      } catch (e: any) {
          addLog(`Import Error: ${e.message}`);
      }
  };

  const addLog = (msg: string) => setLogs(prev => [...prev.slice(-9), `[${new Date().toLocaleTimeString()}] ${msg}`]);


  useEffect(() => {
    import('@neural/engine').then(async (mod) => {
      addLog("Initializing Neural 2.0 Engine...");
      try {
        const engine = new mod.NeuralEngine();
        await engine.init(); 
        engineRef.current = engine;
        setGraphData(engine.getGraphData());
        
        // Subscribe to events
        engine.gpu.subscribe((e: any) => {
            if (e.type === 'loss') setLoss(e.value);
            if (e.type === 'epoch' && Math.random() > 0.9) { 
               // Throttle
            }
        });

        addLog("Engine Ready. WebGPU Compute + Storage Online.");
        setIsReady(true);
      } catch (e: any) {
        addLog(`Error: ${e.message}`);
      }
    });
  }, []);

  const toggleTraining = () => {
      if (trainingRef.current) {
          trainingRef.current = false;
          setIsTraining(false);
          addLog("Training Paused.");
      } else {
          trainingRef.current = true;
          setIsTraining(true);
          addLog("Training Started (Batch Size: 2)...");
          trainLoop();
      }
  };

  const trainLoop = async () => {
      if (!trainingRef.current || !engineRef.current) return;
      
      const gpu = engineRef.current.gpu;
      const size = gpu.networkSize;
      const batch = gpu.batchSize;
      
      const inputs = new Float32Array(size * batch);
      const targets = new Float32Array(size * batch);
      
      // Interactive Demo Data: Logic Gate Training (XOR-ish)
      for (let b = 0; b < batch; b++) {
          const offset = b * size;
          // Random inputs
          const i1 = Math.random() > 0.5 ? 1 : -1;
          const i2 = Math.random() > 0.5 ? 1 : -1;
          
          inputs[offset] = i1;     // Neuron 0
          inputs[offset + 1] = i2; // Neuron 1
          
          // Target (XOR)
          // 1 XOR 1 = -1
          // 1 XOR -1 = 1
          // -1 XOR -1 = -1
          // Tanh Activation -> Target -1 to 1
          const expected = (i1 > 0) !== (i2 > 0) ? 1.0 : -1.0;
          
          targets.fill(-999, offset, offset + size); // Ignore hidden
          targets[offset + size - 1] = expected; // Output Neuron
      }

      await gpu.train(inputs, targets);
      
      if (trainingRef.current) {
          requestAnimationFrame(trainLoop);
      }
  };

  return (
    <main className="flex min-h-screen flex-col items-center relative overflow-hidden selection:bg-brand-primary selection:text-black">
      
      {/* Hero Background (Atmospheric) */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--color-brand-primary-dark)_0%,_transparent_50%)] opacity-20 animate-pulse delay-1000"></div>
         <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom_right,_var(--color-brand-secondary)_0%,_transparent_40%)] opacity-10"></div>
        <div className="absolute inset-0 bg-black/80 backdrop-blur-[1px]"></div>
      </div>

      {/* Content */}
      <div className="z-10 w-full max-w-6xl px-6 py-12 flex flex-col gap-12">
        




  // ... (Render)

        <header className="flex justify-between items-center mb-12">
          {/* ... Title ... */}
          <div className="flex gap-4 items-center">
             <button 
                onClick={toggleListening}
                className={`glass-button px-3 py-1 rounded text-xs font-bold flex items-center gap-2 ${isListening ? 'text-red-400 bg-red-400/10' : 'text-white'}`}
             >
                {isListening ? (
                    <><span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span> Stop Mic</>
                ) : (
                    <><svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path></svg> Start Mic</>
                )}
             </button>
             <div className="h-6 w-px bg-white/20 mx-2"></div>
             <ControlPanel onExport={handleExport} onImport={handleImport} />
             {/* ... */}
             <div className="h-6 w-px bg-white/20 mx-2"></div>
             <button className="glass-button px-4 py-2 rounded-lg text-sm text-white font-medium hover:bg-white/10"
                     onClick={() => window.open('https://github.com/buley/neural', '_blank')}>
               Documentation
             </button>
             <button 
                onClick={toggleTraining}
                disabled={!isReady}
                className={`px-4 py-2 rounded-lg text-sm font-bold transition-all shadow-[0_0_15px_rgba(13,148,136,0.3)] hover:shadow-[0_0_25px_rgba(13,148,136,0.5)] ${
                    !isReady ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed' :
                    isTraining ? 'bg-brand-secondary text-black hover:bg-brand-secondary-light' : 'bg-brand-primary text-black hover:bg-brand-primary-light'
                }`}>
               {isTraining ? 'Stop Training' : 'Start Training'}
             </button>
          </div>
        </header>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Visualization / Engine Status */}
          <div className="lg:col-span-8 flex flex-col gap-8">
            
            {/* The Unmasking of Entropy Visualization Placeholder */}
            <div className="glass-panel rounded-xl p-0 min-h-[500px] flex flex-col justify-center items-center relative overflow-hidden group">
               <Visualizer 
                   data={graphData} 
                   onEdgeClick={async (index) => {
                       if (!engineRef.current || !graphData || !graphData.edges[index]) return;
                       const id = graphData.edges[index].id;
                       addLog(`Cutting synapse connection...`);
                       const newData = await engineRef.current.deleteSynapse(id);
                       setGraphData(newData);
                       addLog(`Synapse lesioned. Network recompiled.`);
                   }}
               />
            </div>

            {/* Terminal / Logs */}
             <div className="glass-panel rounded-xl p-6 h-[250px] overflow-hidden flex flex-col">
                <div className="flex items-center gap-2 mb-4 border-b border-white/10 pb-2">
                  <div className="w-3 h-3 rounded-full bg-red-500/50"></div>
                  <div className="w-3 h-3 rounded-full bg-yellow-500/50"></div>
                  <div className="w-3 h-3 rounded-full bg-green-500/50"></div>
                  <span className="ml-2 font-mono text-xs text-neutral-500">ENGINE_LOGS</span>
                </div>
                <div className="font-mono text-xs space-y-1 overflow-y-auto text-brand-primary/90 scrollbar-hide">
                  {logs.map((log, i) => (
                    <div key={i} className="border-l-2 border-brand-primary/30 pl-2">{log}</div>
                  ))}
                  {!isReady && <div className="animate-pulse pl-2">_</div>}
                  {isReady && <div className="pl-2 text-brand-secondary">Waiting for input...</div>}
                </div>
             </div>

          </div>

          {/* Right Column: Features */}
          <div className="lg:col-span-4 flex flex-col gap-4">
            <ModelZoo onLoad={handleImport} />

            <div className="mt-auto glass-panel p-6 rounded-xl border-brand-secondary/30 bg-brand-secondary/5">
              <h3 className="text-lg font-semibold text-brand-secondary mb-2">Training Status</h3>
              <div className="w-full bg-black/50 h-2 rounded-full overflow-hidden mb-2">
                 <div className="bg-brand-secondary h-full w-[0%]"></div>
              </div>
              <p className="text-xs text-neutral-500">Model untrained.</p>
            </div>
          </div>
        
        </div>

      </div>
    </main>
  );
}
