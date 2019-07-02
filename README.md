## references
<a href='https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c'>An Intuitive Explanation of Connectionist Temporal Classification</a>
<br>
<a href='https://github.com/YCG09/chinese_ocr'>chinese_ocr</a>

## use
edit ```eval.py``` at line ```15```,replace with your image path
```
cd cnn-ctc-text-recognition
python3 -m eval.py
```

## demo
<img src='https://user-images.githubusercontent.com/35487258/60530916-6af3df80-9d2c-11e9-8599-5b6264ea5c3d.jpg'/>
['当成为生态文学的主要']
<img src='https://user-images.githubusercontent.com/35487258/60530995-9c6cab00-9d2c-11e9-89b9-1a2f396f3a38.jpg'/>
['了采用Google搜']

## tensorboard
<img src='https://user-images.githubusercontent.com/35487258/60531611-ef932d80-9d2d-11e9-8295-4099abcd6762.png'/>

## detail
Total stride of cnn network is 8 pixels in width,32 pixels in height


