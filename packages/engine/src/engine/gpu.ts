import shaderCode from './shaders/brain.wgsl?raw';

export class GPUEngine {
    device: GPUDevice | null = null;
    pipeline: GPUComputePipeline | null = null;
    bindGroup: GPUBindGroup | null = null;

    // Buffers
    weightBuffer: GPUBuffer | null = null;
    inputBuffer: GPUBuffer | null = null;
    biasBuffer: GPUBuffer | null = null;
    outputBuffer: GPUBuffer | null = null;
    uniformBuffer: GPUBuffer | null = null;

    networkSize: number = 0;

    async init() {
        if (!navigator.gpu) throw new Error("WebGPU not supported");
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No GPU adapter found");
        this.device = await adapter.requestDevice();

        const shaderModule = this.device.createShaderModule({
            code: shaderCode
        });

        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
        
        console.log("GPUEngine initialized");
    }

    // Prepare buffers based on network size (N)
    prepareBuffers(size: number, weights: Float32Array, biases: Float32Array) {
        if (!this.device || !this.pipeline) throw new Error("GPUEngine not initialized");
        this.networkSize = size;

        // Create Buffers
        this.weightBuffer = this.createBuffer(weights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.inputBuffer = this.createBuffer(new Float32Array(size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.biasBuffer = this.createBuffer(biases, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.outputBuffer = this.createBuffer(new Float32Array(size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        
        // Dimensions Uniform
        const dimArray = new Uint32Array([size]);
        this.uniformBuffer = this.createBuffer(dimArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

        // Bind Group
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.weightBuffer } },
                { binding: 1, resource: { buffer: this.inputBuffer } },
                { binding: 2, resource: { buffer: this.biasBuffer } },
                { binding: 3, resource: { buffer: this.outputBuffer } },
                { binding: 4, resource: { buffer: this.uniformBuffer } },
            ]
        });
    }

    private createBuffer(data: Float32Array | Uint32Array, usage: number): GPUBuffer {
        if (!this.device) throw new Error("Device null");
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: usage,
            mappedAtCreation: true
        });
        if (data instanceof Float32Array) {
            new Float32Array(buffer.getMappedRange()).set(data);
        } else {
            new Uint32Array(buffer.getMappedRange()).set(data);
        }
        buffer.unmap();
        return buffer;
    }

    async runTick(inputs: Float32Array): Promise<Float32Array> {
        if (!this.device || !this.pipeline || !this.bindGroup || !this.inputBuffer || !this.outputBuffer) {
            throw new Error("GPU buffers not ready");
        }

        // Upload Input
        this.device.queue.writeBuffer(this.inputBuffer, 0, inputs);

        // Encode Command
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        
        // Dispatch (Size / WorkgroupSize)
        const workgroupSize = 64;
        const workgroupCount = Math.ceil(this.networkSize / workgroupSize);
        passEncoder.dispatchWorkgroups(workgroupCount);
        passEncoder.end();

        // Read Output
        // We need to copy output buffer to a staging buffer to read it
        // Or simplified: Just read from output if we create a staging buffer. 
        // For performance, we usually keep it on GPU, but for this demo step we read back.
        
        const size = inputs.byteLength;
        const gpuReadBuffer = this.device.createBuffer({
            size: size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        commandEncoder.copyBufferToBuffer(this.outputBuffer, 0, gpuReadBuffer, 0, size);
        
        const gpuCommands = commandEncoder.finish();
        this.device.queue.submit([gpuCommands]);

        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(gpuReadBuffer.getMappedRange());
        const output = new Float32Array(result); // Copy
        gpuReadBuffer.unmap();
        
        return output;
    }
}
