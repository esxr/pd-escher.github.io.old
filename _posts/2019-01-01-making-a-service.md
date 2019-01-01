---
layout: post
date: 2019-01-01
title: "Making a service with Python"
categories: blog
author: Pranav Dhoolia
---
In this task I learn to create and host a service for the clustering work from my [previous post](/blog/2018/12/30/clustering.html) using Python [Flask](http://flask.pocoo.org/).

## Setup a conda environment with Python Flask

```bash
conda create -y -n cluster-service python=3.6
conda activate cluster-service
pip install Flask
```

## Create a Flask Application for clustering

A normal application executes, performs it job, and exits. I want my clustering function to be available all the time. This requires a **server** to be running. A function available on a server is called a **service**. A server may serve several services. The server needs to be told separate paths for each of them. These paths are called **URI**(s). I use [Flask](http://flask.pocoo.org/) module to create my server application. It supports **REST** protocol based binding of services to functions.

The skeleton of my Flask application is as follows.

### Import the necessary classes

```python
from flask import Flask, abort, jsonify, request
```

* `Flask`: has the server functionality
* `abort`: has the functionality to convey erroneous returns
* `jsonify`: transform python objects to json object
* `request`: enables exploration of the service request, parameters and body etc.

### Create Flask server object

```python
app = Flask(__name__)
```

### Link a REST service to my main function

```python
@app.route('/cluster', methods=['POST'])
def cluster():
    pass
```

This is letting the flask `app` know that when it receives a `POST` request for the `URI`: `/cluster` it should invoke the function: `cluster()`

### Start the server

```python
port = os.getenv('PORT', '5000')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(port))
```

This starts the server on the specified `PORT` (or 5000 if not specified). The `if __name__ == '__main__'` condition ensures that the server is started only if this file is directly being executed (and not if like imported in someother file).

## Implement the cluster function

This I do just by organizing the code in the notebook from my [previous post](/blog/2018/12/30/clustering.html) in my Flask application with `cluster()` as the driving function. With that here's the full code of my flask application:

```python
import os
# import Flask and REST classes
from flask import Flask, abort, jsonify, request
# import USE encoder
from use_encoder import USEncoder
# import KMeans clustering algorithm from sklearn
from sklearn.cluster import KMeans
# for some math functions
import math

# Create USE Encoder
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/1"
encoder = USEncoder(USE_MODEL_URL)

# Group texts by cluster labels
def groupby_label(labels, texts):
    texts_by_label = {}
    for label, text in zip(labels,texts):
        label = str(label)
        values = None
        try:
            values = texts_by_label[label]
        except KeyError:
            values = []
            texts_by_label[label] = values        
        values.append(text)
    return texts_by_label

# Create a Flask server app
app = Flask(__name__)

# Specify the URI at which the Flask server app should serve the cluster function
@app.route('/cluster', methods=['POST'])
def cluster():
    # Handle Bad Requests:
    # (requests with no body or texts to cluster in the body are bad)
    if not request.json or "texts" not in request.json:
        abort(400)

    # get texts to cluster from the request body
    texts = request.json["texts"]

    # get k (number of clusters) from the request body
    # (default to 2 if no k specified)
    k = request.json["k"] if "k" in request.json else round(math.sqrt(len(texts)/2))

    # vector encode using USE encoder
    encoded_texts = encoder.encode(texts)
    
    # cluster using k-means
    result = KMeans(n_clusters=k).fit(encoded_texts)

    # group by label
    grouped_by_labels = groupby_label(result.labels_, texts)

    return jsonify(grouped_by_labels), 200

# PORT on which to host the Flask Server
port = os.getenv('PORT', '5000')

# Launch the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(port))
```

## Testing my flask application

My flask application is serving the cluster function as a `REST` service with `POST` method. It is expecting an input with the following structure in the request `body`:

```json
{
    "texts": ['text1', 'text2', '...'],
    "k": 10 // k is optional
}
```

**`curl`** is the most used **command line tool** to test and explore REST services. Here's the curl command I use to test my flask application:

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d@test-input.json \
    http://localhost:5000/cluster
```

* By `-X POST` I am telling it that I am making a `POST` request
* By its two headers (`-H`) I am just telling curl that my request content is a `json` and I expect back a `json` response
* By `-d@test-input.json` I am telling it to pick the data for request `body` from the file `test-input.json`
