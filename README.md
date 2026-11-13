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

## Installation and Requirements

To use the Google ASR module you first need to install the retico-googleasr package:
* Install retico-googleasr:
```pip install retico-googleasr```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-googleasr.git```

### Google Cloud credentials
The module uses the Google Cloud Speech-to-Text API, so you need a **Google Application Credentials JSON file**. Follow the "*Before you begin*" section of [this tutorial](https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries#client-libraries-install-python) to create it, and save the path to that file in the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

## Modules

### `GoogleASRModule`
Subscribes to `AudioIU` and outputs `SpeechRecognitionIU`. The audio is streamed to Google Cloud ASR, and its changing predictions are turned into incremental ADD and REVOKE updates. Words are committed once Google marks the result as final.

If an incoming IU has a `language` attribute (e.g. from [retico-language-detection](https://github.com/retico-team/retico-language-detection)), the module switches to that language automatically.

#### Arguments:
* `language` (str): the language to recognize, as an IETF language tag, e.g. `"en-US"` or `"de-DE"`; defaults to `"en-US"`
* `threshold` (float): the minimum stability of a non-final prediction for it to be published; defaults to `0.8`
* `nchunks` (int): currently unused; defaults to `20`
* `rate` (int): the sample rate of the incoming audio; defaults to `44100`

### Example
**Note:** The example uses `MicrophoneModule` and `CallbackModule` from `retico_core`.

```python
from retico_core import *
from retico_googleasr import *


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


m1 = audio.MicrophoneModule()
m2 = GoogleASRModule("en-US")  # en-US or de-DE or ....
m3 = debug.CallbackModule(callback=callback)

m1.subscribe(m2)
m2.subscribe(m3)

network.run(m1)

print("Running")
input()

network.stop(m1)
```
