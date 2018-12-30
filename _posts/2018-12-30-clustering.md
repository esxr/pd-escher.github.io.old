---
layout: post
date: Dec 30, 2018 
title: Clustering Text with K-Means 
categories: blog
author: Pranav Dhoolia
--- 
I am going to learn and apply clustering techniques. I will use [Health News in
Twitter](https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter) as the
text dataset.

## Plan
I will perform the following tasks.
1. Load the dataset
2. Convert text datapoints to a vector format using **Universal Sentence
Encoder**
3. Cluster the vectorized text using **k-Means**. 
 
## Load the dataset
The [Health News in
Twitter](https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter) data
was collected by [Amir Karami](mailto://karami@sc.edu), University of South
Carolina in 2015 using Twitter API. This dataset contains health news from more
than 15 major health news agencies such as BBC, CNN, and NYT. 


{% highlight python %}
import os
import pandas as pd
{% endhighlight %}


{% highlight python %}
DATA_ROOT='test'
{% endhighlight %}


{% highlight python %}
# directory of `|` separated twitter textfiles
file_list = [os.path.join(DATA_ROOT,file) for file in os.listdir(DATA_ROOT) if file.endswith(".txt")]
df_list = [pd.read_csv(file, header=None, sep='|', usecols=[2]) for file in file_list]
df = pd.concat(df_list)
df.head()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Breast cancer risk test devised http://bbc.in/...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GP workload harming care - BMA poll http://bbc...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Short people's 'heart risk greater' http://bbc...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New approach against HIV 'promising' http://bb...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Coalition 'undermined NHS' - doctors http://bb...</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
texts = df[2].tolist()
{% endhighlight %}
 
## Vectorize text data
Clustering algorithms expect n-dimensional data-points as input. So I first need
to transform my tweets into n-dimensional vectors.
[Word2vec](https://en.wikipedia.org/wiki/Word2vec),
[Text2vec](https://github.com/crownpku/text2vec), Facebook's
[InferSent](https://github.com/facebookresearch/InferSent), and Google's
[Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-
encoder/1) are some of the packages that may be used to get vector embeddings
for text data. I am going to use Google's **Universal Sentence Encoder** here. 
 
### Google Universal Sentence Encoder (USE) 


{% highlight python %}
import tensorflow as tf
import tensorflow_hub as hub
tf.logging.set_verbosity(tf.logging.ERROR)
{% endhighlight %}
 
#### Utility Wrapper on USE
This class eases the usage of USE 


{% highlight python %}
class USEncoder:
    def __init__(self, model_path):
        embed = hub.Module(model_path)
        self.texts = tf.placeholder(dtype=tf.string,shape=[None])
        self.embed_fun = tf.cast(embed(self.texts), tf.float32)
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def encode(self, texts):
        return self.session.run(self.embed_fun, feed_dict={self.texts:texts})
{% endhighlight %}
 
#### Create USE Encoder
One small thing to keep in mind is that USE model is around 1GB in size. On
first use of `hub.Module(..)` the USE model is downloaded locally from
[Tensorflow HUB](https://www.tensorflow.org/hub/). I have setup an environment
variable `TFHUB_CACHE_DIR` with the folder path of where I want to locally store
the model. `hub.Module(..)` uses this environment variable, and if it finds the
cached model, it doesn't download it again. 


{% highlight python %}
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/1"
encoder = USEncoder(USE_MODEL_URL)
{% endhighlight %}
 
#### Vector Embeddings 


{% highlight python %}
encoded_texts = encoder.encode(texts)
{% endhighlight %}
 
# Clustering
Now I am going to apply k-means alongwith some common clustering techniques to
cluster my encoded texts 
 
## k-means
k-means clustering aims to partition `n` observations into `k` clusters in which
each observation belongs to the cluster with the nearest mean. One constraint of
k-means is that it needs to know the value of `k`. In my dataset I don't know
what a reasonable `k` will be. So I am going to try a simple idea. If my data
vectors were in 2 dimensional plane then by a visual observation, I may have
been able to get a rough idea on the number of clusters to expect. But my USE
vectorized data has 512 dimensions. So I am going to use **PCA** (Principal
Component Analysis) to reduce the dimensions to 2 and then get a rough idea. 
 
#### PCA 


{% highlight python %}
from sklearn.decomposition import PCA
{% endhighlight %}


{% highlight python %}
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(encoded_texts)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
{% endhighlight %}
 
#### Visualize 


{% highlight python %}
import matplotlib.pyplot as plt
{% endhighlight %}


{% highlight python %}
plt.figure(figsize=(8,8))
plt.scatter(principalDf['PC1'], principalDf['PC2'], s=0.2)
{% endhighlight %}




    <matplotlib.collections.PathCollection at 0xb6ea37208>



 
![png](/images/clustering_files/clustering_22_1.png) 

 
Well that didn't much clarify as to what `k` should be. So I am now going to try
a bunch of values of `k` and then plot `k` vs. sum of squared errors, to see if
an **elbow analysis** gives me some good sense of the number of clusters. 
 
#### k-means 


{% highlight python %}
from sklearn.cluster import KMeans
{% endhighlight %}


{% highlight python %}
clusters = [KMeans(n_clusters=k).fit(encoded_texts) for k in range(2,40)]
{% endhighlight %}


{% highlight python %}
sse = [cluster.inertia_ for cluster in clusters]
plt.plot(list(range(2,40)), sse)
{% endhighlight %}




    [<matplotlib.lines.Line2D at 0xb6fdb1208>]



 
![png](/images/clustering_files/clustering_27_1.png) 

 
There isn't a clear sense of elbow here. So I am now going to do just one more
thing. I am going to use **KMeans labels to colorize the PCA** scatter plots for
*k* in range 3 - 8. 


{% highlight python %}
plt.figure(figsize=(12,12))

for i in range(1,7):
    k = 2+i
    k_labels = clusters[k-2].labels_
    plt.subplot(320+i)
    plt.scatter(principalDf['PC1'], principalDf['PC2'], s=1.0, c=k_labels)
    plt.title(f'k = {k}')
{% endhighlight %}

 
![png](/images/clustering_files/clustering_29_0.png) 

 
#### Check some text samples from each cluster 


{% highlight python %}
def groupby_label(labels, texts):
    texts_by_label = {}
    for label, text in zip(labels,texts):
        values = None
        try:
            values = texts_by_label[label]
        except KeyError:
            values = []
            texts_by_label[label] = values        
        values.append(text)
    return texts_by_label
{% endhighlight %}
 
**let me sample some texts in the 8 clusters for k=8** 


{% highlight python %}
pd.DataFrame({label:values[:5] 
              for label,values in groupby_label(clusters[8-2].labels_, texts).items()}).T
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Breast cancer risk test devised http://bbc.in/...</td>
      <td>Short people's 'heart risk greater' http://bbc...</td>
      <td>Personal cancer vaccines 'exciting' http://bbc...</td>
      <td>Health highlights http://bbc.in/1EKlFCK</td>
      <td>Drug giant 'blocks' eye treatment http://bbc.i...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GP workload harming care - BMA poll http://bbc...</td>
      <td>Coalition 'undermined NHS' - doctors http://bb...</td>
      <td>Review of case against NHS manager http://bbc....</td>
      <td>VIDEO: NHS: Labour and Tory key policies http:...</td>
      <td>Have GP services got worse? http://bbc.in/1Ci5c22</td>
    </tr>
    <tr>
      <th>7</th>
      <td>New approach against HIV 'promising' http://bb...</td>
      <td>80,000 'might die' in future outbreak http://b...</td>
      <td>Skin cancer 'linked to holiday boom' http://bb...</td>
      <td>Unsafe food 'growing global threat' http://bbc...</td>
      <td>Uganda circumcision truck fights HIV http://bb...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>VIDEO: 'All day is empty, what am I going to d...</td>
      <td>VIDEO: 'Overhaul needed' for end-of-life care ...</td>
      <td>VIDEO: Skin cancer spike 'from 60s holidays' h...</td>
      <td>VIDEO: Welcome to the designer asylum http://b...</td>
      <td>VIDEO: Why are we having less sex? http://bbc....</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Care for dying 'needs overhaul' http://bbc.in/...</td>
      <td>A&amp;amp;E waiting hits new worst level http://bb...</td>
      <td>Parties row over GP opening hours http://bbc.i...</td>
      <td>The Bolivian women who knit parts for hearts h...</td>
      <td>First Europe non-beating heart swap http://bbc...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Why strenuous runs may not be so bad after all...</td>
      <td>Parents rarely spot child obesity http://bbc.i...</td>
      <td>Office workers 'too sedentary' http://bbc.in/1...</td>
      <td>Fitness linked to lower cancer risk http://bbc...</td>
      <td>Approach to obesity 'inexplicable' http://bbc....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VIDEO: Health surcharge for non-EU patients ht...</td>
      <td>Child heart surgery deaths 'halved' http://bbc...</td>
      <td>Ambulance progress 'not fast enough' http://bb...</td>
      <td>Care system 'gets biggest shake-up' http://bbc...</td>
      <td>More veterans seek mental health aid http://bb...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MS drug 'may already be out there' http://bbc....</td>
      <td>E-cigarette use 'high among teens' http://bbc....</td>
      <td>VIDEO: Could cannabis oil cure cancer? http://...</td>
      <td>VIDEO: Toxic smog prompts health warning http:...</td>
      <td>New global fund for tobacco control http://bbc...</td>
    </tr>
  </tbody>
</table>
</div>


 
## Conclusion
This concludes my exploration of clustering some text with k-means. I learnt
several things:
* loading and slicing text dataset with pandas
* text vectorization techniques (in particular Google Universal Sentence
Encoder)
* Reducing data dimensions with Principal Component Analysis
* Visualizing data with matplotlib
* k-means clustering
* getting a sense of k with elbow analysis
* visualizing clusters with colorized scatter plots
* checking the quality of clusters by sampling texts

I sense that the quality of clusters wasn't quite upto the mark. I guess a bit
of **data-cleaning** would help (e.g. removing things like URL links from the
tweets).

I am now looking forward to explore more advanced clustering techniques like
**Hierarchical Agglomerative Clustering**, and **Affinity Propagation**! 
