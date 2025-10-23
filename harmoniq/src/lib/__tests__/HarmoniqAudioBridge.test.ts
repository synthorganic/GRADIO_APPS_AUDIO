import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import { HarmoniqAudioBridge } from "../HarmoniqAudioBridge";
import type { DeckPlaybackDiagnostics } from "../../types";

class MockAudioNode {
  connections: MockAudioNode[] = [];

  connect(node: MockAudioNode) {
    this.connections.push(node);
    return node;
  }

  disconnect() {
    this.connections = [];
  }
}

class MockGainNode extends MockAudioNode {
  gain = {
    value: 1,
    setTargetAtTime: vi.fn(),
    setValueAtTime: vi.fn((value: number) => {
      this.gain.value = value;
    }),
    linearRampToValueAtTime: vi.fn((value: number) => {
      this.gain.value = value;
    }),
    cancelScheduledValues: vi.fn(),
  };
}

class MockAnalyserNode extends MockAudioNode {
  fftSize = 512;
  smoothingTimeConstant = 0.85;
  private data = new Float32Array(this.fftSize);

  getFloatTimeDomainData(target: Float32Array) {
    if (target.length !== this.data.length) {
      target.set(new Float32Array(target.length));
      return;
    }
    target.set(this.data);
  }
}

class MockBiquadFilterNode extends MockAudioNode {
  type: string = "lowpass";
  frequency = { value: 0 };
  Q = { value: 1 };
  gain = { value: 0 };
}

class MockAudioBuffer {
  constructor(public duration: number, public sampleRate = 48000, public numberOfChannels = 2) {}

  get length() {
    return Math.floor(this.duration * this.sampleRate);
  }
}

class MockAudioBufferSourceNode extends MockAudioNode {
  buffer: MockAudioBuffer | null = null;
  onended: (() => void) | null = null;
  startedOffsets: number[] = [];
  playbackRate = {
    value: 1,
    setTargetAtTime: vi.fn(),
    setValueAtTime: vi.fn(),
  };

  start(_when: number, offset = 0) {
    this.startedOffsets.push(offset);
  }

  stop() {
    if (this.onended) {
      this.onended();
    }
  }
}

class MockAudioContext {
  currentTime = 0;
  sampleRate = 48000;
  destination = new MockAudioNode();
  decodeDuration = 0;
  closed = false;
  createdSources: MockAudioBufferSourceNode[] = [];
  createdGains: MockGainNode[] = [];

  createGain() {
    const gain = new MockGainNode();
    this.createdGains.push(gain);
    return gain;
  }

  createAnalyser() {
    return new MockAnalyserNode();
  }

  createBiquadFilter() {
    return new MockBiquadFilterNode();
  }

  createBufferSource() {
    const source = new MockAudioBufferSourceNode();
    this.createdSources.push(source);
    return source;
  }

  decodeAudioData(_data: ArrayBuffer) {
    return Promise.resolve(new MockAudioBuffer(this.decodeDuration, this.sampleRate));
  }

  close = vi.fn(async () => {
    this.closed = true;
  });
}

describe("HarmoniqAudioBridge", () => {
  let context: MockAudioContext;
  let bridge: HarmoniqAudioBridge;

  beforeEach(() => {
    vi.useFakeTimers();
    context = new MockAudioContext();
    context.decodeDuration = 12.4;
    bridge = new HarmoniqAudioBridge({
      context: context as unknown as AudioContext,
      debug: false,
      meterIntervalMs: 5,
    });
  });

  afterEach(() => {
    bridge.dispose();
    vi.useRealTimers();
  });

  it("loads, plays, seeks, reloads, and emits diagnostics", async () => {
    const buffer = new ArrayBuffer(16);
    const analysis = await bridge.analyzeSource({
      id: "analysis",
      arrayBuffer: buffer,
      objectUrl: "blob:analysis",
      name: "Analysis",
    });
    expect(analysis.durationSeconds).toBeCloseTo(12.4);

    const snapshots: DeckPlaybackDiagnostics[] = [];
    const unsubscribe = bridge.subscribeDiagnostics((snapshot) => snapshots.push(snapshot));

    await bridge.loadDeckAudio("A", {
      id: "deck-track",
      arrayBuffer: buffer,
      objectUrl: "blob:deck",
      name: "Deck",
    });
    expect(snapshots.at(-1)).toMatchObject({
      deckId: "A",
      isPlaying: false,
      durationSeconds: 12.4,
    });

    snapshots.length = 0;
    await bridge.playDeck("A");
    expect(snapshots.at(-1)?.isPlaying).toBe(true);

    snapshots.length = 0;
    context.currentTime = 1.25;
    vi.advanceTimersByTime(20);
    expect(snapshots.at(-1)?.currentTimeSeconds ?? 0).toBeGreaterThan(1.2);

    snapshots.length = 0;
    await bridge.seekDeck("A", 4.5);
    const seekSnapshot = snapshots.at(-1);
    expect(seekSnapshot?.deckId).toBe("A");
    expect(seekSnapshot?.currentTimeSeconds ?? 0).toBeCloseTo(4.5, 3);

    snapshots.length = 0;
    await bridge.stopDeck("A");
    expect(snapshots.at(-1)?.isPlaying).toBe(false);

    snapshots.length = 0;
    context.decodeDuration = 9.8;
    await bridge.reloadDeck("A", {
      id: "deck-track",
      arrayBuffer: buffer,
      objectUrl: "blob:deck",
      name: "Deck",
    });
    const reloadSnapshot = snapshots.at(-1);
    expect(reloadSnapshot?.deckId).toBe("A");
    expect(reloadSnapshot?.durationSeconds ?? 0).toBeCloseTo(9.8, 3);
    expect(reloadSnapshot?.isPlaying).toBe(false);

    unsubscribe();
  });

  it("ramps deck input gain on play and stop", async () => {
    const buffer = new ArrayBuffer(8);
    await bridge.loadDeckAudio("A", {
      id: "deck-track",
      arrayBuffer: buffer,
      objectUrl: "blob:deck",
      name: "Deck",
    });

    await bridge.playDeck("A");
    const fadeInGain = context.createdGains.find((gain) =>
      gain.gain.linearRampToValueAtTime.mock.calls.some(([value]) => value === 1),
    );
    expect(fadeInGain).toBeDefined();

    await bridge.stopDeck("A");
    const fadeOutGain = context.createdGains.find((gain) =>
      gain.gain.linearRampToValueAtTime.mock.calls.some(([value]) => value === 0),
    );
    expect(fadeOutGain).toBeDefined();
  });

  it("honors custom fade durations for playback toggles", async () => {
    const buffer = new ArrayBuffer(8);
    await bridge.loadDeckAudio("A", {
      id: "deck-track",
      arrayBuffer: buffer,
      objectUrl: "blob:deck",
      name: "Deck",
    });

    context.currentTime = 0.75;
    await bridge.playDeck("A", { fadeDurationSeconds: 1.25 });
    const fadeInGain = context.createdGains.find((gain) =>
      gain.gain.linearRampToValueAtTime.mock.calls.some(([value]) => value === 1),
    );
    expect(fadeInGain).toBeDefined();
    const fadeInCall = fadeInGain?.gain.linearRampToValueAtTime.mock.calls.find(([value]) => value === 1);
    expect(fadeInCall).toBeDefined();
    expect(fadeInCall?.[1]).toBeCloseTo(0.75 + 1.25, 6);

    context.currentTime = 2.4;
    await bridge.stopDeck("A", { fadeDurationSeconds: 0.4 });
    const fadeOutGain = context.createdGains.find((gain) =>
      gain.gain.linearRampToValueAtTime.mock.calls.some(([value]) => value === 0),
    );
    expect(fadeOutGain).toBeDefined();
    const fadeOutCall = fadeOutGain?.gain.linearRampToValueAtTime.mock.calls.find(([value]) => value === 0);
    expect(fadeOutCall).toBeDefined();
    expect(fadeOutCall?.[1]).toBeCloseTo(2.4 + 0.4, 6);
  });

  it("updates playback rate when timestretch changes", async () => {
    const buffer = new ArrayBuffer(32);
    const snapshots: DeckPlaybackDiagnostics[] = [];
    const unsubscribe = bridge.subscribeDiagnostics((snapshot) => snapshots.push(snapshot));

    await bridge.loadDeckAudio("A", {
      id: "deck-track",
      arrayBuffer: buffer,
      objectUrl: "blob:deck",
      name: "Deck",
    });

    await bridge.playDeck("A");
    const source = context.createdSources.at(-1);
    expect(source?.playbackRate.value).toBeCloseTo(1);

    snapshots.length = 0;
    context.currentTime = 1;
    vi.advanceTimersByTime(10);
    const beforeChange = snapshots.at(-1)?.currentTimeSeconds ?? 0;

    context.currentTime = 1.5;
    bridge.setTimestretch(1.5);
    expect(source?.playbackRate.setTargetAtTime).toHaveBeenCalledWith(1.5, 1.5, 0.05);

    snapshots.length = 0;
    context.currentTime = 2;
    vi.advanceTimersByTime(10);
    const afterChange = snapshots.at(-1)?.currentTimeSeconds ?? 0;
    expect(afterChange).toBeGreaterThan(beforeChange + 0.7);

    unsubscribe();
  });
});
