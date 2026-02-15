export class AudioInput {
  ctx: AudioContext;
  analyser: AnalyserNode | null = null;
  source: MediaStreamAudioSourceNode | null = null;
  dataArray: Uint8Array | null = null;
  isListening = false;

  constructor() {
    this.ctx = new (window.AudioContext ||
      (window as any).webkitAudioContext)();
  }

  async start() {
    if (this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 64; // Low resolution for neural input (32 bins)
      this.source = this.ctx.createMediaStreamSource(stream);
      this.source.connect(this.analyser);

      const bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(bufferLength);
      this.isListening = true;
    } catch (err) {
      console.error('Error accessing microphone:', err);
      throw err;
    }
  }

  stop() {
    if (this.source) {
      this.source.disconnect();
      this.source.mediaStream.getTracks().forEach((track) => track.stop());
      this.source = null;
    }
    this.isListening = false;
  }

  // Returns normalized frequency data (0.0 to 1.0)
  getFrequencyData(): Float32Array {
    if (!this.analyser || !this.dataArray) return new Float32Array(0);

    this.analyser.getByteFrequencyData(this.dataArray);

    // Normalize 0-255 to -1.0 to 1.0 (for tanh inputs)
    const normalized = new Float32Array(this.dataArray.length);
    for (let i = 0; i < this.dataArray.length; i++) {
      normalized[i] = this.dataArray[i] / 128.0 - 1.0;
    }
    return normalized;
  }
}
