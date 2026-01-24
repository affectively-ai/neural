
// Minimal WebNN Type Definitions for TypeScript
// Based on W3C Web Neural Network API Draft

interface MLContext {
    compute(graph: MLGraph, inputs: Record<string, ArrayBufferView>, outputs: Record<string, ArrayBufferView>): Promise<MLComputeResult>;
}

interface MLComputeResult {
    inputs: Record<string, ArrayBufferView>;
    outputs: Record<string, ArrayBufferView>;
}

interface MLGraphBuilder {
    input(name: string, descriptor: MLOperandDescriptor): MLOperand;
    constant(descriptor: MLOperandDescriptor, buffer: ArrayBufferView): MLOperand;
    matmul(a: MLOperand, b: MLOperand): MLOperand;
    add(a: MLOperand, b: MLOperand): MLOperand;
    tanh(x: MLOperand): MLOperand;
    build(outputs: Record<string, MLOperand>): Promise<MLGraph>;
}

interface MLGraph {}

interface MLOperand {}

interface MLOperandDescriptor {
    dataType: 'float32' | 'float16' | 'int32' | 'uint32' | 'int8' | 'uint8';
    dimensions: number[];
}

interface MLContextOptions {
    deviceType?: 'cpu' | 'gpu' | 'npu';
    powerPreference?: 'default' | 'high-performance' | 'low-power';
}

interface NavigatorML {
    createContext(options?: MLContextOptions): Promise<MLContext>;
}

interface Navigator {
    ml?: NavigatorML;
}
