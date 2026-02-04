"""
A Module that offers different types of real time speech recognition.
"""
import queue
import threading
import retico_core
from retico_core.text import SpeechRecognitionIU
from retico_core.audio import AudioIU
from google.cloud import speech as gspeech

# The Google ASR API uses IETF language tags when the language recognition module outputs them in ISO 639-1 format
MAP_ISO_TO_IETF = { # To be expanded at will
    "ar": "ar-EG",
    "bg": "bg-BG",
    "ca": "ca-ES",
    "cs": "cs-CZ",
    "da": "da-DK",
    "de": "de-DE",
    "el": "el-GR",
    "en": "en-US",
    "es": "es-ES",
    "et": "et-EE",
    "fi": "fi-FI",
    "fr": "fr-FR",
    "he": "he-IL",
    "hi": "hi-IN",
    "hu": "hu-HU",
    "id": "id-ID",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "lt": "lt-LT",
    "lv": "lv-LV",
    "ms": "ms-MY",
    "nl": "nl-NL",
    "no": "no-NO",
    "pl": "pl-PL",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "sk": "sk-SK",
    "sl": "sl-SI",
    "sv": "sv-SE",
    "tr": "tr-TR",
    "uk": "uk-UA",
    "vi": "vi-VN",
    "zh": "zh-CN",
}

class GoogleASRModule(retico_core.AbstractModule):
    """A Module that recognizes speech by utilizing the Google Speech API."""

    def __init__(
        self,
        language: str = "en-US",
        threshold: float = 0.8,
        nchunks: int = 20,
        rate: int = 44100,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.iso_language = "en"
        self.language = language

        self.nchunks = nchunks
        self.rate = rate

        self.client = None
        self.streaming_config = None
        self.responses = []

        self.threshold = threshold
        
        self.audio_buffer = queue.Queue()
        
        self.latest_input_iu = None

        self._recognition_thread = None
        self._is_running = False
        
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
            
            if hasattr(iu, "language") and iu.language != self.iso_language:
                new_iso = iu.language
                if new_iso in MAP_ISO_TO_IETF:
                    previous = self.language
                    self.iso_language = new_iso
                    try:
                        self.language = MAP_ISO_TO_IETF[new_iso]
                        print(f"[GoogleASRModule] Language changed from {previous} to {self.language}.")
                        self._restart_recognition()
                    except KeyError:
                        print(f"[GoogleASRModule] Language {new_iso} not supported by Google ASR. Keeping {previous}.")

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
        try:
            for response in self.responses:
                predictions, text, stability, confidence, final = self._extract_results(response)
                if not predictions or (stability < self.threshold and confidence == 0.0 and not final):
                    continue
                current_text = text

                um, new_tokens = retico_core.text.get_text_increment(self, current_text)

                if len(new_tokens) == 0 and final:
                    for iu in self.current_output:
                        um.add_iu(iu, retico_core.UpdateType.COMMIT)
                    self.current_output = []

                for i, token in enumerate(new_tokens):
                    output_iu = self.create_iu(self.latest_input_iu)
                    eou = final and (i == len(new_tokens) - 1)
                    output_iu.set_asr_results(predictions, token, 0.0, 0.99, eou)
                    
                    self.current_output.append(output_iu)
                    um.add_iu(output_iu, retico_core.UpdateType.ADD)
                    
                    if eou:
                        for iu in self.current_output:
                            um.add_iu(iu, retico_core.UpdateType.COMMIT)
                        self.current_output = []

                self.latest_input_iu = None
                self.append(um)

        except Exception as e:
            print(f"[GoogleASRModule] Exception in prediction loop: {e}")

    def _restart_recognition(self):
        """Stop, reconfigure and restart the background recognition thread."""
        self._is_running = False
        self.audio_buffer.put(None)  # will break out of generator

        if self._recognition_thread and self._recognition_thread.is_alive():
            self._recognition_thread.join()

        self.setup()
        self._is_running = True
        self._recognition_thread = threading.Thread(
            target=self._produce_predictions_loop, daemon=True
        )
        self._recognition_thread.start()

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
        self._is_running = True
        self._recognition_thread = threading.Thread(
            target=self._produce_predictions_loop, daemon=True
        )
        self._recognition_thread.start()

    def shutdown(self):
        self._is_running = False
        self.audio_buffer.put(None)
        if self._recognition_thread and self._recognition_thread.is_alive():
            self._recognition_thread.join()