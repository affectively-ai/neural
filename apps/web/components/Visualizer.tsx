'use client';

import { useMemo, useRef, useLayoutEffect, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Text } from '@react-three/drei';
import * as THREE from 'three';
import { computeLayout, GraphEdge } from '../lib/layout';

interface VisualizerProps {
  data?: {
    nodeCount: number;
    edges: GraphEdge[];
    nodes?: { id: string; type: string }[];
  };
  onNodeClick?: (index: number) => void;
  onEdgeClick?: (index: number) => void;
}

function NeuronInstances({
  positions,
  count,
  nodes,
}: {
  positions: Float32Array;
  count: number;
  nodes?: { type: string }[];
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const color = new THREE.Color();
  const tempObject = new THREE.Object3D();

  useLayoutEffect(() => {
    if (!meshRef.current) return;

    for (let i = 0; i < count; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];

      tempObject.position.set(x, y, z);
      tempObject.scale.set(1, 1, 1);
      tempObject.updateMatrix();
      meshRef.current.setMatrixAt(i, tempObject.matrix);

      // Color based on type
      const type = nodes?.[i]?.type || 'hidden';
      if (type === 'input') color.set('#fb7185'); // Red/Pink
      else if (type === 'output') color.set('#0d9488'); // Teal
      else if (type === 'cloud') color.set('#ffffff'); // White/Cloud
      else color.set('#334155'); // Slate/Hidden

      meshRef.current.setColorAt(i, color);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor)
      meshRef.current.instanceColor.needsUpdate = true;
  }, [positions, count, nodes]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial roughness={0.4} metalness={0.6} />
    </instancedMesh>
  );
}

function SynapseInstances({
  positions,
  edges,
  onEdgeClick,
}: {
  positions: Float32Array;
  edges: GraphEdge[];
  onEdgeClick?: (i: number) => void;
}) {
  const ref = useRef<THREE.LineSegments>(null);
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    // 2 points per edge
    const pos = new Float32Array(edges.length * 2 * 3);

    // We will update this in useLayoutEffect/useFrame
    // But initial alloc is needed
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    return geo;
  }, [edges.length]);

  useLayoutEffect(() => {
    if (!ref.current) return;

    const posAttr = ref.current.geometry.getAttribute(
      'position'
    ) as THREE.BufferAttribute;
    const array = posAttr.array as Float32Array;

    for (let i = 0; i < edges.length; i++) {
      const u = edges[i].source;
      const v = edges[i].target;

      array[i * 6] = positions[u * 3];
      array[i * 6 + 1] = positions[u * 3 + 1];
      array[i * 6 + 2] = positions[u * 3 + 2];

      array[i * 6 + 3] = positions[v * 3];
      array[i * 6 + 4] = positions[v * 3 + 1];
      array[i * 6 + 5] = positions[v * 3 + 2];
    }
    posAttr.needsUpdate = true;
  }, [positions, edges]);

  return (
    <lineSegments
      ref={ref}
      geometry={geometry}
      onClick={(e) => {
        e.stopPropagation();
        // Raycast gives us the index of the face/segment?
        if (onEdgeClick && e.index !== undefined) {
          onEdgeClick(Math.floor(e.index / 2));
        }
      }}
    >
      <lineBasicMaterial
        color="#475569"
        transparent
        opacity={0.3}
        linewidth={1}
      />
    </lineSegments>
  );
}

export default function Visualizer({ data, onEdgeClick }: VisualizerProps) {
  // Generate Layout
  const { positions, edges, count, nodes } = useMemo(() => {
    if (!data || data.nodeCount === 0)
      return { positions: new Float32Array(0), edges: [], count: 0, nodes: [] };

    const positions = computeLayout(data.nodeCount, data.edges, 50);
    return {
      positions,
      edges: data.edges,
      count: data.nodeCount,
      nodes: data.nodes,
    };
  }, [data]);

  return (
    <div className="w-full h-full min-h-[500px] bg-black rounded-xl overflow-hidden relative">
      <Canvas
        camera={{ position: [0, 0, 40], fov: 50 }}
        raycaster={{ params: { Line: { threshold: 0.5 } } as any }}
      >
        <color attach="background" args={['#050505']} />
        <fog attach="fog" args={['#050505', 30, 60]} />

        <ambientLight intensity={1.5} />
        <pointLight position={[10, 10, 10]} intensity={2} color="#fb7185" />
        <pointLight position={[-10, -10, -10]} intensity={2} color="#0d9488" />

        <Stars
          radius={100}
          depth={50}
          count={5000}
          factor={4}
          saturation={0}
          fade
          speed={1}
        />

        {count > 0 && (
          <NeuronInstances positions={positions} count={count} nodes={nodes} />
        )}
        {count > 0 && (
          <SynapseInstances
            positions={positions}
            edges={edges}
            onEdgeClick={onEdgeClick}
          />
        )}

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={true}
          autoRotateSpeed={0.5}
        />
      </Canvas>

      <div className="absolute bottom-4 left-4 pointer-events-none">
        <div className="text-xs text-brand-primary font-mono">
          NODES: {count}
          <br />
          EDGES: {edges.length}
          <br />
          RENDERER: WEBGL
        </div>
      </div>
    </div>
  );
}
