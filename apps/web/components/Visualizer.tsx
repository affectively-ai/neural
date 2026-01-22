// ... (Imports)

interface VisualizerProps {
    data?: {
        nodeCount: number;
        edges: GraphEdge[];
    };
    onNodeClick?: (index: number) => void;
    onEdgeClick?: (index: number) => void;
}

// ... (Subcomponents)

export default function Visualizer({ data, onEdgeClick }: VisualizerProps) {
    // Generate Layout
    const { positions, edges, count } = useMemo(() => {
        if (!data || data.nodeCount === 0) return { positions: new Float32Array(0), edges: [], count: 0 };
        
        const positions = computeLayout(data.nodeCount, data.edges, 50);
        return { positions, edges: data.edges, count: data.nodeCount };
    }, [data]);
    
    // ... (Render)
    return (
        <div className="w-full h-full min-h-[500px] bg-black rounded-xl overflow-hidden relative">
            <Canvas camera={{ position: [0, 0, 40], fov: 50 }} raycaster={{ params: { Line: { threshold: 0.5 } } }}>
                <color attach="background" args={['#050505']} />
                <fog attach="fog" args={['#050505', 30, 60]} />
                
                <ambientLight intensity={1.5} />
                <pointLight position={[10, 10, 10]} intensity={2} color="#fb7185" />
                <pointLight position={[-10, -10, -10]} intensity={2} color="#0d9488" />
                
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
                
                {count > 0 && <NeuronInstances positions={positions} count={count} />}
                {count > 0 && <SynapseInstances positions={positions} edges={edges} onEdgeClick={onEdgeClick} />}
                
                <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} autoRotate={true} autoRotateSpeed={0.5} />
            </Canvas>
            
             <div className="absolute bottom-4 left-4 pointer-events-none">
                <div className="text-xs text-brand-primary font-mono">
                    NODES: {count}<br/>
                    EDGES: {edges.length}<br/>
                    RENDERER: WEBGL
                </div>
            </div>
        </div>
    );
}
