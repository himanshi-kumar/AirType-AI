"""
audio.py — Sprint 7: Sound Feedback Engine

WHY SOUND FEEDBACK?
--------------------
Physical keyboards give tactile + auditory feedback for every keypress.
Without feedback, touchless air-typing feels unresponsive — the user
can't tell if a gesture registered or was missed.

Sound bridges this gap: each gesture type gets a distinct audio signature,
letting the user build muscle memory for the keyboard without looking at
the typed text bar.

WHY SYNTHESIZE INSTEAD OF LOADING FILES?
-----------------------------------------
We could ship .wav or .mp3 files in an assets/ directory.
Instead, we synthesize all sounds in-memory using NumPy:

  ADVANTAGES
  ----------
  1. Zero assets: no files to ship, no paths to manage, no file I/O.
  2. Parameterizable: frequency, duration, volume are all adjustable at runtime.
  3. Reproducible: same formula → same sound, regardless of filesystem.
  4. Educational: building sounds from math reinforces signal processing concepts.

  DISADVANTAGES
  --------------
  1. Sounds are simpler (sine/sweep, not sampled).
  2. Cannot replicate recorded keyboard click sounds exactly.

HOW DIGITAL AUDIO WORKS
-------------------------
Digital audio is a sequence of floating-point numbers (samples) representing
air pressure over time. At 44100 Hz (CD quality), 44100 samples = 1 second.

A sine wave at frequency f:
    y[t] = amplitude × sin(2π × f × t)

Where t = [0, 1/44100, 2/44100, ...] — time in seconds.

The human hearing range is 20 Hz – 20,000 Hz.
  Low pitch (bass):  C2 = 65 Hz
  Middle C:         C4 = 261 Hz
  Keyboard clicks:  A4 = 440 Hz (standard pitch)

WHY NON-BLOCKING PLAYBACK?
----------------------------
sounddevice.play(array, blocking=False) returns IMMEDIATELY.
The OS audio driver plays the sound asynchronously.
A 50ms click sound would otherwise pause the 30fps camera loop for ~1.5 frames.
Non-blocking playback means zero frame drops.
"""

import numpy as np
import sounddevice as sd


# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100    # Hz — CD quality; 44,100 samples per second
CHANNELS    = 1        # Mono output (one speaker channel is enough for feedback)


class SoundPlayer:
    """
    Synthesizes and plays short audio feedback sounds for keyboard events.

    SINGLE RESPONSIBILITY
    ----------------------
    SoundPlayer answers: "Play a sound for this keyboard event."
    It does NOT know about keyboards, gestures, or TTS.
    It only operates on NumPy arrays and audio hardware.

    DESIGN PATTERN: PRE-COMPUTED SOUNDS
    -------------------------------------
    All sounds are synthesized once in __init__ and stored as NumPy arrays.
    When a key is pressed, we just call sd.play(self._click_sound) — no
    computation happens in the real-time loop.

    This avoids any latency spike mid-frame. The 30fps loop only calls
    sd.play(), which is a single system call (essentially free).
    """

    def __init__(self, volume: float = 0.4):
        """
        Pre-synthesize all sounds at initialization time.

        Parameters
        ----------
        volume : float
            Master volume from 0.0 (mute) to 1.0 (max). Default: 0.4.
            Moderate volume so feedback doesn't overpower the TTS voice.
        """
        self.volume = volume

        # Pre-compute all sounds once (not per-keypress)
        self._keypress_sound   = self._sine(freq=440,  duration=0.05)   # 50ms, A4
        self._space_sound      = self._sine(freq=220,  duration=0.08)   # 80ms, A3
        self._backspace_sound  = self._sweep(freq_start=330, freq_end=165, duration=0.07)  # descending
        self._suggestion_sound = self._sweep(freq_start=523, freq_end=784, duration=0.12)  # rising C5→G5
        self._speak_sound      = self._chord(freqs=[440, 550], duration=0.15)              # 440+550Hz
        self._clear_sound      = self._chord(freqs=[587, 440, 293], duration=0.18)         # Sprint 12: D5+A4+D4 descending chord
        self._mode_sound       = self._chord(freqs=[523, 659], duration=0.10)              # Sprint 12: C5+E5 chime

        # Apply fade-out envelope to all sounds to avoid clicks/pops
        # at the end of each sample (abrupt cutoff creates a "click" artifact)
        for attr in ("_keypress_sound", "_space_sound", "_backspace_sound",
                     "_suggestion_sound", "_speak_sound", "_clear_sound", "_mode_sound"):
            setattr(self, attr, self._apply_fade(getattr(self, attr)))

    # ── Sound Synthesis ────────────────────────────────────────────────────────

    def _sine(self, freq: float, duration: float) -> np.ndarray:
        """
        Generate a pure sine wave.

        FORMULA
        --------
        y[t] = volume × sin(2π × freq × t)

        Parameters
        ----------
        freq     : float   Frequency in Hz (e.g. 440 = A4, concert pitch)
        duration : float   Duration in seconds (e.g. 0.05 = 50ms)

        Returns
        -------
        np.ndarray   float32 array of audio samples, values in [-volume, volume]
        """
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        return (self.volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _sweep(self, freq_start: float, freq_end: float, duration: float) -> np.ndarray:
        """
        Generate a linear frequency sweep (chirp) from freq_start to freq_end.

        A sweep slides through frequencies, giving a "rising" or "falling"
        tonal characteristic. Great for backspace (falling = "undo") and
        suggestions (rising = "completion" reward).

        FORMULA
        --------
        The instantaneous frequency changes linearly:
            f(t) = freq_start + (freq_end - freq_start) × (t / duration)

        Integrating to get phase:
            phase(t) = 2π × (freq_start × t + (freq_end - freq_start) × t² / (2 × duration))

        Parameters
        ----------
        freq_start : float   Starting frequency in Hz
        freq_end   : float   Ending frequency in Hz
        duration   : float   Duration in seconds

        Returns
        -------
        np.ndarray   float32 array of audio samples
        """
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        # Accumulated phase via integration of instantaneous frequency
        phase = 2 * np.pi * (freq_start * t + (freq_end - freq_start) * t**2 / (2 * duration))
        return (self.volume * np.sin(phase)).astype(np.float32)

    def _chord(self, freqs: list, duration: float) -> np.ndarray:
        """
        Generate a chord by overlaying multiple sine waves and normalizing.

        A chord is the sum of individual sine waves, one per frequency.
        After summing, we re-normalize to prevent clipping (values > 1.0 are
        distorted by hardware).

        WHY NORMALIZE?
        ---------------
        Two sine waves summed have amplitude up to 2×. sounddevice clips
        values outside [-1.0, 1.0]. Dividing by len(freqs) keeps us safe.

        Parameters
        ----------
        freqs    : list[float]   List of frequencies in Hz
        duration : float         Duration in seconds

        Returns
        -------
        np.ndarray   float32 array of audio samples
        """
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
        # Normalize so max amplitude = volume (not len(freqs) × volume)
        signal = signal / len(freqs)
        return (self.volume * signal).astype(np.float32)

    def _apply_fade(self, samples: np.ndarray, fade_ms: float = 10.0) -> np.ndarray:
        """
        Apply a linear fade-out to the tail of a sound to eliminate end-click artifacts.

        WHY THIS MATTERS
        -----------------
        When audio playback stops abruptly at a non-zero sample value,
        the speaker membrane jumps discontinuously — producing an audible
        "click" or "pop". A short linear fade to zero eliminates this.

        We only fade the tail (10ms) to preserve the attack (the onset
        click that provides the tactile confirmation).

        Parameters
        ----------
        samples : np.ndarray   Input audio array
        fade_ms : float        Length of fade in milliseconds. Default: 10ms.

        Returns
        -------
        np.ndarray   Same array with tail faded to silence
        """
        fade_samples = int(SAMPLE_RATE * fade_ms / 1000)
        fade_samples = min(fade_samples, len(samples))
        fade_curve = np.linspace(1.0, 0.0, fade_samples)
        result = samples.copy()
        result[-fade_samples:] *= fade_curve
        return result

    # ── Public Playback API ────────────────────────────────────────────────────

    def play_keypress(self) -> None:
        """
        Play a short click for regular letter key presses.
        440Hz (A4), 50ms — classic typewriter click.
        """
        sd.play(self._keypress_sound, samplerate=SAMPLE_RATE)

    def play_space(self) -> None:
        """
        Play a deeper thud for the SPACE key.
        220Hz (A3), 80ms — lower pitch matches the "heavier" spacebar action.
        """
        sd.play(self._space_sound, samplerate=SAMPLE_RATE)

    def play_backspace(self) -> None:
        """
        Play a descending blip for the BACK (backspace) key.
        330→165Hz sweep, 70ms — falling pitch reinforces the "undo" concept.
        """
        sd.play(self._backspace_sound, samplerate=SAMPLE_RATE)

    def play_suggestion(self) -> None:
        """
        Play a rising chime when a suggestion word is selected.
        523→784Hz (C5→G5) sweep, 120ms — rising pitch = reward/completion.
        """
        sd.play(self._suggestion_sound, samplerate=SAMPLE_RATE)

    def play_speak(self) -> None:
        """
        Play an activation chord when the SPEAK key is pinched.
        440+550Hz chord, 150ms — richer than a click to signal a major action.
        """
        sd.play(self._speak_sound, samplerate=SAMPLE_RATE)

    def play_clear(self) -> None:
        """
        Play a descending chord when the CLR (clear) key is pinched.
        D5+A4+D4 chord, 180ms — distinct reset audio signature.
        """
        sd.play(self._clear_sound, samplerate=SAMPLE_RATE)

    def play_mode_switch(self) -> None:
        """
        Play a crisp dual-tone chime when toggling between ABC and 123 layouts.
        523+659Hz (C5+E5) chord, 100ms.
        """
        sd.play(self._mode_sound, samplerate=SAMPLE_RATE)

