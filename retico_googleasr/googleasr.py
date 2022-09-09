"""
A Module that offers different types of real time speech recognition.
"""

import queue
import threading
import retico_core
from retico_core.text import SpeechRecognitionIU
from retico_core.audio import AudioIU
from google.cloud import speech as gspeech


class GoogleASRModule(retico_core.AbstractModule):
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
            if ut != retico_core.UpdateType.ADD:
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
                um = retico_core.UpdateMessage()
                if s < self.threshold and c == 0.0 and not f:
                    continue
                current_text = t

                um, new_tokens = retico_core.text.get_text_increment(self, current_text)

                if len(new_tokens) == 0:
                    if not f:
                        continue
                    else:
                        output_iu = self.create_iu(self.latest_input_iu)
                        output_iu.set_asr_results(p, "", s, c, f)
                        output_iu.committed = True
                        self.current_output = []
                        um.add_iu(output_iu, retico_core.UpdateType.ADD)

                for i, token in enumerate(new_tokens):
                    output_iu = self.create_iu(self.latest_input_iu)
                    eou = f and i == len(new_tokens) - 1
                    output_iu.set_asr_results(p, token, 0.0, 0.99, eou)
                    if eou:
                        output_iu.committed = True
                        self.current_output = []
                    else:
                        self.current_output.append(output_iu)
                    um.add_iu(output_iu, retico_core.UpdateType.ADD)

                self.latest_input_iu = None
                self.append(um)

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
