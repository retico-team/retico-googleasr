# retico-googleasr

This project contains the incremental module for running Google Cloud ASR in a retico
environment. Google ASR provides continuous speech recognition updates by updating the
current prediction. As this is not in the correct format for incremental processing, the
recognized speech gets incrementalized and only updates to the previous states are
published. E.g.,

```
[UpdateType]        [text]             [stability]            [final]
UpdateType.ADD:     this               (0.8999999761581421) - False
UpdateType.ADD:     isn't a sentence   (0.8999999761581421) - False
UpdateType.REVOKE:  isn't a sentence   (0.8999999761581421) - False
UpdateType.ADD:     is a test sentence (0.0)                - True
```

Google Speech-To-Text also provides different transcription models for speech recognition that can be found [here](https://docs.cloud.google.com/speech-to-text/docs/transcription-model).

## Installing

To use the Automatic Speech Recognition module utilizing google cloud speech, you need to install the third-party package first.

For this, you may follow the first two steps of [this tutorial](https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries#client-libraries-install-python).

Important is, that you create your **Google Application Credentials json file** and save the path to that file into the global variable `GOOGLE_APPLICATION_CREDENTIALS` (look for the "*Before you begin*" section on the tutorial page).

After that you can install the package with

```bash
$ pip install git+https://github.com/retico-team/retico-googleasr
```

## GoogleASR Example

```python
from retico_core.audio import MicrophoneModule
from retico_core.debug import CallbackModule
from retico_core.abstract import UpdateType
from retico_googleasr import GoogleASRModule


msg = []


def callback(update_msg):
    global msg
    for x, ut in update_msg:
        if ut == UpdateType.ADD:
            msg.append(x)
        if ut == UpdateType.REVOKE:
            msg.remove(x)
    txt = ""
    committed = False
    for x in msg:
        txt += x.text + " "
        committed = committed or x.committed
    print(" " * 80, end="\r")
    print(f"{txt}", end="\r")
    if committed:
        msg = []
        print("")


mic = MicrophoneModule()
asr = GoogleASRModule(language="en-US", rate=16_000, model="chirp_3")  # en-US or de-DE or ....
callback = CallbackModule(callback=callback)

mic.subscribe(asr)
asr.subscribe(callback)

mic.run()
asr.run()
callback.run()

print("Running")
input()

mic.stop()
asr.stop()
callback.stop()
```