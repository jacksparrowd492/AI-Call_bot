"""Audio helpers: PCM <-> mu-law conversion and 20 ms chunking for Twilio Media Streams.

Twilio Media Streams audio facts:
- format: audio/x-mulaw, 8000 Hz, 1 channel
- one 20 ms frame = 160 mu-law bytes (160 samples = 320 PCM bytes), base64-encoded
"""
import audioop

SAMPLE_RATE = 8000
FRAME_MS = 20
SAMPLES_PER_FRAME = 160                 # 20 ms at 8 kHz
FRAME_BYTES = SAMPLES_PER_FRAME         # mu-law: 1 byte per sample


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Decode Twilio inbound mu-law to 16-bit little-endian PCM (for logging/backup)."""
    return audioop.ulaw2lin(mulaw_bytes, 2)


def pcm16_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Encode 16-bit PCM (8 kHz mono) to mu-law. Input must be even-length."""
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    return audioop.lin2ulaw(pcm_bytes, 2)


class PcmToMulawFramer:
    """Incremental PCM -> 20 ms mu-law frame converter with a single PCM buffer."""

    def __init__(self):
        self._buf = b""                 # unaligned PCM bytes (may end mid-sample)

    def feed(self, pcm_bytes: bytes):
        """Feed raw PCM bytes, return a list of complete 160-byte mu-law frames."""
        buf = self._buf + pcm_bytes
        usable = len(buf) - (len(buf) % 2)          # drop odd tail byte for now
        aligned = buf[:usable]
        self._buf = buf[usable:]

        n_samples = usable // 2
        n_frames = n_samples // SAMPLES_PER_FRAME
        if n_frames == 0:
            return []
        consumed_samples = n_frames * SAMPLES_PER_FRAME
        consumed_bytes = consumed_samples * 2

        mulaw = audioop.lin2ulaw(aligned[:consumed_bytes], 2)
        frames = [mulaw[i * FRAME_BYTES:(i + 1) * FRAME_BYTES] for i in range(n_frames)]
        # Keep the whole-sample remainder for the next feed
        self._buf = aligned[consumed_bytes:] + self._buf
        return frames

    def flush(self):
        """Return any final partial frame (zero-padded to a full 20 ms) and reset."""
        out = b""
        buf = self._buf
        if len(buf) % 2:
            buf = buf[:-1]
        if buf:
            samples = len(buf) // 2
            pad = SAMPLES_PER_FRAME - (samples % SAMPLES_PER_FRAME)
            if pad == SAMPLES_PER_FRAME:
                pad = 0
            buf = buf + b"\x00\x00" * pad
            out = audioop.lin2ulaw(buf, 2)
        self.__init__()
        return out
