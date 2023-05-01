Download Link: https://assignmentchef.com/product/solved-bia660-assignment6-clustering-and-topic-modeling
<br>
In this assignment, you’ll need to use the following dataset:

text_train.json: This file contains a list of documents. It’s used for training models text_test.json: This file contains a list of documents and their ground-truth labels. It’s used for testing performance. This file is in the format shown below. Note, each document has a list of labels. You can load these files using json.load()

<table width="0">

 <tbody>

  <tr>

   <td width="349"><strong>Text</strong></td>

   <td width="427"><strong>Labels</strong></td>

  </tr>

  <tr>

   <td width="349">paraglider collides with hot air balloon …</td>

   <td width="427">[‘Disaster and Accident’, ‘Travel &amp; Transportation’]</td>

  </tr>

  <tr>

   <td width="349">faa issues fire warning for lithium …</td>

   <td width="427">[‘Travel &amp; Transportation’]</td>

  </tr>

  <tr>

   <td width="349">….</td>

   <td width="427">…</td>

  </tr>

 </tbody>

</table>

<h1>Q1: K-Mean Clustering</h1>

Define a function <strong>cluster_kmean()</strong> as follows:

Take two file name strings as inputs: <em>train</em>_<em>file</em> is the file path of text_train.json, and <em>test</em>_<em>file</em> is the file path of text_test.json

Use <strong>KMeans</strong> to cluster documents in <em>train</em>_<em>file</em> into 3 clusters by <strong>cosine similarity </strong>Test the clustering model performance using <em>test</em>_<em>file</em>:

Predict the cluster ID for each document in <em>test</em>_<em>file</em>.

Let’s only use the <strong>first label</strong> in the ground-truth label list of each test document, e.g. for the first document in the table above, you set the ground_truth label to “Disaster and Accident” only.

Apply <strong>majority vote</strong> rule to dynamically map the predicted cluster IDs to the ground-truth labels in <em>test</em>_<em>file</em>. <strong>Be sure not to hardcode the mapping</strong> (e.g. write code like {0: “Disaster and Accident”}), because a cluster may corrspond to a different topic in each run.

Calculate <strong>precision/recall/f-score</strong> for each label

This function has no return. Print out confusion matrix, precision/recall/f-score.

<h1>Q2: LDA Clustering</h1>

Define a function <strong>cluster_lda()</strong> as follows:

Take two file name strings as inputs: <em>train</em>_<em>file</em> is the file path of text_train.json, and <em>test</em>_<em>file</em> is the file path of text_test.json

Use <strong>LDA</strong> to train a topic model with documents in <em>train</em>_<em>file</em> and the number of topics <em>K</em> = 3 Predict the topic distribution of each document in <em>test</em>_<em>file</em>, and select <strong>only the topic with highest probability</strong> as the predicted topic

Evaluates the topic model performance as follows:

Similar to Q1, let’s use the <strong>first label</strong> in the label list of <em>test</em>_<em>file</em> as the ground_truth label.

Apply <strong>majority vote rule</strong> to map the topics to the labels.

Calculate <strong>precision/recall/f-score</strong> for each label and print out precision/recall/f-score. Return topic distribution and the original ground-truth labels of each document in <em>test</em>_<em>file </em>Also, provide a document which contains:

performance comparison between Q1 and Q2 describe how you tune the model parameters, e.g. min_df, alpha, max_iter etc.

<h1>Q3 (Bonus): Overlapping Clustering</h1>

In Q2, you predict one label for each document in <em>test</em>_<em>file</em>. In this question, try to discover multiple labels if appropriate. Define a function <strong>overlapping_cluster</strong> as follows:

Take the outputs of Q2 (i.e. topic distribution and the labels of each document in <em>test</em>_<em>file</em>) as inputs

Set a threshold for each topic (i.e. <em>TH </em>= [<em>th</em>0, <em>th</em>1, <em>th</em>2]). A document is predicted to belong to a topic <em>i</em> only if the topic probability &gt; <em>th</em><em>i</em> for <em>i </em>∈ [0, 1, 2].

The threshold is determined as follows:

Vary the threshold for each topic from 0.05 to 0.95 with an increase of 0.05 in each round to evalute the topic model performance:

Apply <strong>majority vote rule</strong> to map the predicted topics to the ground-truth labels in <em>test</em>_<em>file</em>

Calculate <strong>f1-score</strong> for each label

For each label, pick the threshold value which maximizes the f1-score

Return the threshold and f1-score of each label

In [145]:

<strong>from</strong> <strong>sklearn.feature_extraction.text</strong> <strong>import</strong> CountVectorizer <strong>from</strong> <strong>nltk.cluster</strong> <strong>import</strong> KMeansClusterer, cosine_distance <strong>from</strong> <strong>sklearn.decomposition</strong> <strong>import</strong> LatentDirichletAllocation

<em># add more</em>

In [146]:

actual_class  Disaster and Accident  News and Economy  Travel &amp; Tran sportation

cluster                                                              0                                70                 0                135

<ul>

 <li>130 7                8</li>

 <li>10 199</li>

</ul>

41

Cluster 0: Topic Travel &amp; Transportation

Cluster 1: Topic Disaster and Accident

Cluster 2: Topic News and Economy                          precision    recall  f1-score   support

Disaster and Accident       0.90      0.62      0.73       210

News and Economy       0.80      0.97      0.87       206 Travel &amp; Transportation       0.66      0.73      0.69       184

micro avg       0.77      0.77      0.77       600               macro avg       0.78      0.77      0.77       600            weighted avg       0.79      0.77      0.77       600

iteration: 1 of max_iter: 25 iteration: 2 of max_iter: 25 iteration: 3 of max_iter: 25 iteration: 4 of max_iter: 25 iteration: 5 of max_iter: 25 iteration: 6 of max_iter: 25 iteration: 7 of max_iter: 25 iteration: 8 of max_iter: 25 iteration: 9 of max_iter: 25 iteration: 10 of max_iter: 25 iteration: 11 of max_iter: 25 iteration: 12 of max_iter: 25 iteration: 13 of max_iter: 25 iteration: 14 of max_iter: 25 iteration: 15 of max_iter: 25 iteration: 16 of max_iter: 25 iteration: 17 of max_iter: 25 iteration: 18 of max_iter: 25 iteration: 19 of max_iter: 25 iteration: 20 of max_iter: 25 iteration: 21 of max_iter: 25 iteration: 22 of max_iter: 25 iteration: 23 of max_iter: 25 iteration: 24 of max_iter: 25 iteration: 25 of max_iter: 25

actual_class  Disaster and Accident  News and Economy  Travel &amp; Tran sportation

cluster                                                              0                                30                18                138

<ul>

 <li>12 182                8</li>

 <li>168 6</li>

</ul>

38

Cluster 0: Topic Travel &amp; Transportation

Cluster 1: Topic News and Economy

Cluster 2: Topic Disaster and Accident

precision    recall  f1-score   support

Disaster and Accident       0.79      0.80      0.80       210

News and Economy       0.90      0.88      0.89       206 Travel &amp; Transportation       0.74      0.75      0.75       184

micro avg       0.81      0.81      0.81       600               macro avg       0.81      0.81      0.81       600            weighted avg       0.81      0.81      0.81       600

Disaster and Accident      0.45

News and Economy           0.55 Travel &amp; Transportation    0.30 dtype: float64

Disaster and Accident      0.798122

News and Economy           0.888889 Travel &amp; Transportation    0.773218