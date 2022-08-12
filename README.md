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

## Installing

To use the Automatic Speech Recognition module utilizing google cloud speech, you need to install the third-party package first.

For this, you may follow the first two steps of [this tutorial](https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries#client-libraries-install-python).

Important is, that you create your **Google Application Credentials json file** and save the path to that file into the global variable `GOOGLE_APPLICATION_CREDENTIALS` (look for the "*Before you begin*" section on the tutorial page).

After that you can install the package with

```bash
$ pip install retico-googleasr
```

## Documentation

Sadly, there is no proper documentation for retico-googleasr right now, but you can 
start using the GoogleASRModule like this:

```python
from retico_core import *
from retico_googleasr import *


def callback(update_msg):
    for x, ut in update_msg:
        print(f"{ut}: {x.text} ({x.stability}) - {x.final}")


m1 = audio.MicrophoneModule(5000)
m2 = GoogleASRModule("en-US")  # en-US or de-DE or ....
m3 = debug.CallbackModule(callback=callback)

m1.subscribe(m2)
m2.subscribe(m3)

m1.run()
m2.run()
m3.run()

input()

m1.stop()
m2.stop()
m3.stop()
```