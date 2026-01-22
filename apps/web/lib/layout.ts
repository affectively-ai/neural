
export interface GraphNode {
    id: string;
    index: number;
}

export interface GraphEdge {
    source: number; // Index
    target: number; // Index
    weight: number;
}

export function computeLayout(nodeCount: number, edges: GraphEdge[], iterations = 50): Float32Array {
    const positions = new Float32Array(nodeCount * 3);
    const velocities = new Float32Array(nodeCount * 3);
    
    // 1. Initialize Random Positions
    for (let i = 0; i < nodeCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 50;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 50;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 50;
    }

    // Constants
    const k = 2.0; // Ideal length
    const repulsion = 10.0;
    const attraction = 0.1;
    const centerPull = 0.05;
    const dt = 0.1;
    const damping = 0.85;

    // 2. Simulation Loop (Simple CPU implementation)
    // For 1000 nodes, O(N^2) repulsion is 1M ops per tick. 50 ticks = 50M ops. 
    // This might block the main thread for ~50-100ms. Acceptable for initialization.
    for (let iter = 0; iter < iterations; iter++) {
        
        // Repulsion (All pairs) - Naive O(N^2)
        // Optimization: Only check nearby? Or just random sampling? 
        // Let's do full check but optimized inner loop.
        for (let i = 0; i < nodeCount; i++) {
            const ix = positions[i * 3];
            const iy = positions[i * 3 + 1];
            const iz = positions[i * 3 + 2];
            
            let fx = 0, fy = 0, fz = 0;

            for (let j = 0; j < nodeCount; j++) {
                if (i === j) continue;
                
                const jx = positions[j * 3];
                const jy = positions[j * 3 + 1];
                const jz = positions[j * 3 + 2];
                
                const dx = ix - jx;
                const dy = iy - jy;
                const dz = iz - jz;
                
                const distSq = dx*dx + dy*dy + dz*dz + 0.01;
                const dist = Math.sqrt(distSq);
                
                // F = k^2 / dist
                const force = repulsion / distSq;
                
                fx += (dx / dist) * force;
                fy += (dy / dist) * force;
                fz += (dz / dist) * force;
            }
            
            // Central Gravity
            fx -= ix * centerPull;
            fy -= iy * centerPull;
            fz -= iz * centerPull;

            velocities[i * 3] += fx * dt;
            velocities[i * 3 + 1] += fy * dt;
            velocities[i * 3 + 2] += fz * dt;
        }

        // Attraction (Edges)
        for (const edge of edges) {
            const u = edge.source;
            const v = edge.target;
            
            const ux = positions[u * 3];
            const uy = positions[u * 3 + 1];
            const uz = positions[u * 3 + 2];
            
            const vx = positions[v * 3];
            const vy = positions[v * 3 + 1];
            const vz = positions[v * 3 + 2];
            
            const dx = vx - ux;
            const dy = vy - uy;
            const dz = vz - uz;
            
            const dist = Math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01;
            
            // F = dist^2 / k
            const force = (dist * dist) / k * attraction;
            
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            const fz = (dz / dist) * force;
            
            velocities[u * 3] += fx * dt;
            velocities[u * 3 + 1] += fy * dt;
            velocities[u * 3 + 2] += fz * dt;
            
            velocities[v * 3] -= fx * dt;
            velocities[v * 3 + 1] -= fy * dt;
            velocities[v * 3 + 2] -= fz * dt;
        }

        // Update Positions
        for (let i = 0; i < nodeCount; i++) {
            positions[i * 3] += velocities[i * 3] * dt;
            positions[i * 3 + 1] += velocities[i * 3 + 1] * dt;
            positions[i * 3 + 2] += velocities[i * 3 + 2] * dt;
            
            // Damping
            velocities[i * 3] *= damping;
            velocities[i * 3 + 1] *= damping;
            velocities[i * 3 + 2] *= damping;
        }
    }
    
    return positions;
}
