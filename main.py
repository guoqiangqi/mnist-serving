import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

from mnist_inference import inference
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("/mnist-serving/test_data", one_hot=True)
batch = data.test.next_batch(1)

# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = batch[0] 
    im = Image.fromarray((batch[0].reshape(28,28)*255).astype(np.uint8))
    im.save("/mnist-serving/ckpt/input.jpeg")
    #print(np.array(request.json))
    webIm = Image.fromarray(255 - np.array(request.json).astype(np.uint8).reshape(28,28))
    webIm.save("/mnist-serving/ckpt/webIm.jpeg")

    print(batch[1])
    # output1 = regression(input)
    output2 = inference(input)
    return jsonify(results=[output2, output2])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9005)
