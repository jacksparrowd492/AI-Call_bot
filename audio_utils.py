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


# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------
# The problem this solves: Deepgram transcribes whatever it is given. On a call
# from a street, an office or a room with a fan, the background never stops, so
# the recogniser keeps producing short "turns" that were never speech - and the
# bot answers them. Confidence thresholds only see the transcript, by which
# point the damage is done; the fix has to land on the AUDIO.
#
# Two rules shape the implementation:
#
#   1. Never delete frames. Deepgram works out that a turn has ended by
#      listening to the silence after it, so the stream must stay continuous
#      and in real time. Noise frames are ATTENUATED, not dropped.
#   2. Never gate on an absolute number alone. One caller's line hiss is
#      another caller's whisper, so the floor is measured per call, from the
#      quiet moments of that call, and the threshold follows it.
#
# The gate is deliberately forgiving: it opens on the first loud frame, holds
# open through the gaps between words, and when shut it attenuates rather than
# silences - so a caller the gate misjudges is quieter, never missing.

MULAW_SILENCE = 0xFF                    # mu-law encoding of zero

# The floor cannot wander off to either extreme: too low and every rustle is
# "loud", too high and a soft caller is treated as noise.
_MIN_FLOOR_RMS = 30.0
_MAX_FLOOR_RMS = 1200.0

# How the measured floor follows the line, per 20 ms frame. It tracks the
# MINIMUM, not the average: it drops quickly to whatever quiet the line
# actually has, and creeps back up at 0.2% a frame (about +10% a second).
#
# That asymmetry is the whole trick. Speech is loud and lasts seconds, so it
# can only nudge the floor - the gate cannot deafen itself part-way through a
# long sentence. A line that is genuinely noisier from the start still gets
# learned, over tens of seconds, instead of the gate quietly giving up: an
# earlier version only measured the floor while the gate was SHUT, so a call
# whose background never dropped below the opening threshold never measured
# anything at all and the gate stayed open for the whole call.
_FALL = 0.5                             # gets quieter: follow it down at once
_CREEP = 1.002                          # gets louder: creep, do not chase


class NoiseGate:
    """Adaptive noise gate over Twilio's inbound mu-law frames.

    Feed it every inbound frame with `process()`; it returns a frame of the
    same length and format, with background-only frames turned down.
    """

    def __init__(self, ratio=2.5, floor_rms=200, hangover_ms=320,
                 attenuation=0.12, enabled=True, frame_ms=FRAME_MS):
        self.enabled = enabled
        self.ratio = max(1.0, float(ratio))
        self.floor_rms = max(0.0, float(floor_rms))
        self.attenuation = min(1.0, max(0.0, float(attenuation)))
        self.hangover_frames = max(1, int(hangover_ms / max(1, frame_ms)))

        # Gain moves this much per frame, so the gate fades instead of
        # clicking - a click is a transient, and transients look like
        # consonants to a recogniser.
        self._attack = 1.0                        # open instantly
        self._release = 1.0 / 6                   # close over ~120 ms

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Forget the measured floor. Called when the line conditions change -
        after the bot has been speaking, for instance."""
        self._floor = None
        self._hold = 0
        self._gain = 1.0
        self.open_frames = 0
        self.total_frames = 0

    @property
    def is_open(self) -> bool:
        return self._hold > 0

    @property
    def noise_floor(self) -> float:
        return self._floor or 0.0

    def threshold(self) -> float:
        floor = self._floor if self._floor is not None else self.floor_rms
        return max(self.floor_rms, floor * self.ratio)

    # ------------------------------------------------------------------ audio
    def process(self, mulaw: bytes) -> bytes:
        """Return the frame with background noise attenuated."""
        if not self.enabled or not mulaw:
            return mulaw

        try:
            pcm = audioop.ulaw2lin(mulaw, 2)
            rms = audioop.rms(pcm, 2)
        except Exception:
            # Never let the gate be the reason a call loses audio.
            return mulaw

        self.total_frames += 1

        if rms >= self.threshold():
            self._hold = self.hangover_frames
        elif self._hold > 0:
            self._hold -= 1

        if self._hold:
            self.open_frames += 1
        self._track(rms)

        target = 1.0 if self._hold else self.attenuation
        if target > self._gain:
            self._gain = min(target, self._gain + self._attack)
        elif target < self._gain:
            self._gain = max(target, self._gain - self._release)

        if self._gain >= 0.999:
            return mulaw
        if self._gain <= 0.0:
            return bytes([MULAW_SILENCE]) * len(mulaw)

        try:
            return audioop.lin2ulaw(audioop.mul(pcm, 2, self._gain), 2)
        except Exception:
            return mulaw

    def _track(self, rms: float):
        if self._floor is None:
            self._floor = float(rms)
        elif rms < self._floor:
            self._floor += (rms - self._floor) * _FALL
        else:
            self._floor = self._floor * _CREEP + 0.5
        self._floor = max(_MIN_FLOOR_RMS, min(_MAX_FLOOR_RMS, self._floor))


def gate_from_settings():
    """Build the gate the way config.py says. Kept here so callers do not have
    to know the settings names."""
    from config import settings
    return NoiseGate(
        ratio=settings.noise_gate_ratio,
        floor_rms=settings.noise_gate_floor_rms,
        hangover_ms=settings.noise_gate_hangover_ms,
        attenuation=settings.noise_gate_attenuation,
        enabled=settings.noise_gate,
    )
