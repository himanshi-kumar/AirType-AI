"""
speech.py — Sprint 6: Text-to-Speech Engine

WHAT IS TEXT-TO-SPEECH (TTS)?
------------------------------
TTS converts written text into spoken audio output.
When the user pinches the SPEAK key, AirType reads the typed sentence aloud.

This is the same technology powering:
  - Screen readers (accessibility tools for visually impaired users)
  - Voice assistants (Siri, Google Assistant, Alexa)
  - Navigation apps ("Turn left in 500 meters")

WHY pyttsx3?
--------------
pyttsx3 is a Python TTS library that works OFFLINE using the OS's native engine:
  - macOS  → NSSpeechSynthesizer (built into every Mac)
  - Windows → SAPI5 (built into every Windows PC)
  - Linux  → espeak (commonly pre-installed)

Advantages over cloud TTS (Google Cloud, AWS Polly):
  - Zero latency (no network request)
  - Works offline (no internet required)
  - Free (no API key or billing)
  - Privacy (audio never leaves the device)

WHY A BACKGROUND THREAD?
---------------------------
pyttsx3's runAndWait() BLOCKS the calling thread until speech finishes.
If we call it on the main thread, the camera feed freezes for 2-3 seconds
while "HELLO WORLD" is being spoken. At 30fps, that's 60-90 dropped frames.

Solution: run speech on a daemon thread.
  - Main thread: camera capture + ML inference + keyboard drawing (30fps)
  - Background thread: TTS engine speaks text (1-5 seconds, no rush)

DAEMON THREAD
--------------
daemon=True means the thread dies automatically when the main program exits.
Without this, if the user presses 'q' to quit while speech is playing,
the program would hang until speech finishes. Daemon threads prevent that.

THREADING LOCK
---------------
pyttsx3's engine is NOT thread-safe. If two threads call engine.say()
simultaneously, it crashes. A Lock ensures only one thread uses the engine
at any time. This is called mutual exclusion (mutex).
"""

import threading
import pyttsx3


class Speaker:
    """
    Non-blocking text-to-speech engine.

    SINGLE RESPONSIBILITY
    ----------------------
    Speaker answers one question: "Say this text out loud."
    It does NOT know about keyboards, frames, or gestures.
    It only operates on strings and audio output.

    ATTRIBUTES
    ----------
    _engine : pyttsx3.Engine
        The TTS engine instance. Initialized once in __init__.
    _lock : threading.Lock
        Prevents concurrent access to the engine (not thread-safe).
    _speaking : bool
        True while speech is in progress. Used by is_speaking property.
    _thread : threading.Thread | None
        The current background speech thread, if any.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0):
        """
        Initialize the TTS engine with voice settings.

        Parameters
        ----------
        rate : int
            Speech speed in words per minute. Default: 175.
            - 100 = slow, deliberate speech (good for accessibility)
            - 175 = natural conversational speed
            - 250 = fast, news-anchor speed
            pyttsx3 default is 200, but 175 sounds more natural.

        volume : float
            Output volume from 0.0 (mute) to 1.0 (max). Default: 1.0.
        """
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        self._lock = threading.Lock()
        self._speaking = False
        self._thread = None

    def speak(self, text: str) -> None:
        """
        Speak the given text on a background thread (non-blocking).

        If speech is already in progress, it is stopped first and the
        new text replaces it. This ensures the user always hears the
        most recently typed sentence.

        WHY NON-BLOCKING?
        -------------------
        The main loop runs at 30fps. If speak() blocked for 3 seconds,
        the camera feed would freeze. By running on a background thread,
        the main loop continues rendering frames while speech plays.

        Parameters
        ----------
        text : str   The text to speak aloud. Empty text is silently ignored.
        """
        if not text or not text.strip():
            return

        # Stop any in-progress speech before starting new
        self.stop()

        # Launch speech on a daemon thread
        self._thread = threading.Thread(
            target=self._speak_blocking,
            args=(text,),
            daemon=True,
        )
        self._thread.start()

    def _speak_blocking(self, text: str) -> None:
        """
        Internal: speak text synchronously (blocks until done).

        This runs on the background thread, NOT the main thread.
        The Lock prevents two threads from using the engine simultaneously.

        WHY A SEPARATE METHOD?
        -----------------------
        threading.Thread(target=...) requires a callable. We can't inline
        the lock + engine logic in speak() because speak() must return
        immediately (non-blocking). This method contains the blocking part.
        """
        with self._lock:
            self._speaking = True
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except RuntimeError:
                # Engine may raise if stop() was called during speech.
                # This is expected and safe to ignore.
                pass
            finally:
                self._speaking = False

    def stop(self) -> None:
        """
        Stop any in-progress speech immediately.

        Called before starting new speech (to prevent overlap)
        and during cleanup when the program exits.

        pyttsx3's stop() is safe to call even when nothing is playing.
        """
        with self._lock:
            try:
                self._engine.stop()
            except RuntimeError:
                pass
            self._speaking = False

    @property
    def is_speaking(self) -> bool:
        """
        Read-only property: is speech currently in progress?

        Can be used by the UI to show a visual indicator (e.g. a speaker icon
        or pulsing animation) while the engine is speaking.
        """
        return self._speaking
