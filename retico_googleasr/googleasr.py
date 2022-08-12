"""
A Module that offers different types of real time speech recognition.
"""

import queue
import threading
from retico_core import *
from retico_core.text import SpeechRecognitionIU
from retico_core.audio import AudioIU
from google.cloud import speech as gspeech


class GoogleASRModule(AbstractModule):
    """A Module that recognizes speech by utilizing the Google Speech API."""

    def __init__(
        self, language="en-US", threshold=0.8, nchunks=20, rate=44100, **kwargs
    ):
        """Initialize the GoogleASRModule with the given arguments.

        Args:
            language (str): The language code the recognizer should use.
            threshold (float): The amount of stability needed to forward an update.
            nchunks (int): Number of chunks that should trigger a new
                prediction.
            rate (int): The framerate of the input audio
        """
        super().__init__(**kwargs)
        self.language = language
        self.nchunks = nchunks
        self.rate = rate

        self.client = None
        self.streaming_config = None
        self.responses = []

        self.threshold = threshold

        self.audio_buffer = queue.Queue()

        self.latest_input_iu = None

    @staticmethod
    def name():
        return "Google ASR Module"

    @staticmethod
    def description():
        return "A Module that incrementally recognizes speech."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != UpdateType.ADD:
                continue
            self.audio_buffer.put(iu.raw_audio)
            if not self.latest_input_iu:
                self.latest_input_iu = iu
        return None

    @staticmethod
    def _extract_results(response):
        predictions = []
        text = None
        stability = 0.0
        confidence = 0.0
        final = False
        for result in response.results:
            if not result or not result.alternatives:
                continue

            if not text:
                final = result.is_final
                stability = result.stability
                text = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence
            predictions.append(
                (
                    result.alternatives[0].transcript,
                    result.stability,
                    result.alternatives[0].confidence,
                    result.is_final,
                )
            )
        return predictions, text, stability, confidence, final

    def _generator(self):
        while self._is_running:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self.audio_buffer.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self.audio_buffer.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

    def _produce_predictions_loop(self):
        requests = (
            gspeech.StreamingRecognizeRequest(audio_content=content)
            for content in self._generator()
        )
        self.responses = self.client.streaming_recognize(
            self.streaming_config, requests
        )
        for response in self.responses:
            p, t, s, c, f = self._extract_results(response)
            if p:
                um = UpdateMessage()
                if s < self.threshold and c == 0.0 and not f:
                    continue
                current_text = t
                if self.current_ius:
                    um, current_text = self.get_increment(current_text)
                if current_text.strip() == "" and not f:
                    continue

                output_iu = self.create_iu(self.latest_input_iu)
                self.latest_input_iu = None
                output_iu.set_asr_results(p, current_text, s, c, f)
                self.current_ius.append(output_iu)
                if output_iu.final:
                    self.current_ius = []
                    output_iu.committed = True
                um.add_iu(output_iu, UpdateType.ADD)
                self.append(um)

    def get_increment(self, new_text):
        """Compares the full text given by the asr with the IUs that are already
        produced and returns only the increment from the last update. It revokes all
        previously produced IUs that do not match."""
        um = UpdateMessage()
        for iu in self.current_ius:
            if new_text.startswith(iu.text):
                new_text = new_text[len(iu.text) :]
            else:
                iu.revoked = True
                um.add_iu(iu, UpdateType.REVOKE)
        self.current_ius = [iu for iu in self.current_ius if not iu.revoked]
        return um, new_text

    def setup(self):
        self.client = gspeech.SpeechClient()
        config = gspeech.RecognitionConfig(
            encoding=gspeech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.rate,
            language_code=self.language,
        )
        self.streaming_config = gspeech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

    def prepare_run(self):
        t = threading.Thread(target=self._produce_predictions_loop)
        t.start()

    def shutdown(self):
        self.audio_buffer.put(None)
