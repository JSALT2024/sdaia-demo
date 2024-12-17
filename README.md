
# sdaia-demo
**Sign Language Recognition - Alphanumerals in SAUDI SL**

This demo is a proof of concept for the recognition system of the SAUDI SL. It uses MediaPipe and DINO in the backend, trained on the SAUDI SL dataset from Mohammad Alghannami and Maram Aljuaid.

## How to run

 1. Clone repository
 ```commandline
git clone https://github.com/JSALT2024/sdaia-demo.git
```
    

 2. Install dependencies

```commandline
import os
os.chdir('sdaia-demo/')
pip install -r requirements.txt
```

 3. Run the demo
 ```commandline
python saudiSLGradio.py
```
 4. Open http://127.0.0.1:7860/

 5. You are good to go! Upload any image or use your webcam 📷

![](https://github.com/JSALT2024/sdaia-demo/blob/main/img/demo.jpg)
