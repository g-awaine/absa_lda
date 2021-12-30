# Aspect-Based Sentiment Analysis

Customer review analysis allows companies to understand customers’ needs and preferences, and improve their services or create new products based accordingly. This can help businesses with better customer retention and acquisition, and hence revenue growth.
Customer reviews are often multidimensional. For example, a restaurant customer could have a positive view on waiters and servers but might be less satisfied with the menu and the variety of choices offered by the restaurant. 
Aspect-based sentiment analysis allows to separately analyze each category, which corresponds to a specific component of the services/products. This method is also helpful in rapid sorting of customer complaints and assigning customer support tasks, which means effective customer support.
The review analysis pipeline for this project included the following steps:

1. Extracting topics/aspects of a review
2. Sentiment analysis for each aspect
3. Extracting the subject of the review and its descriptors

![figure1](/figures/fig1.png)

Here is an example of a review analysis:
Review: ‘Atmosphere is chill and cool. Hostess was extremely accommodating when we arrived early for our reservation.’

![figure2](/figures/fig2.png)

<br>

## Dataset
The dataset comes from restaurant visitors reviews and included 3149, and 400 labelled reviews in the train set and test sets, respectively. Each review is labeled from a total of 8 aspects: 'food', 'menu', 'service', 'place', 'price', ‘miscellaneous', 'staff', ‘ambience’. The original data, in xml format, was loaded and converted to a data frame with one-hot encoded presentation of aspects.[link to data preparation notebook](https://github.com/MahsaShokouhi/Aspect_Based_Sentiment_Analysis/blob/master/notebooks/data_preparation.ipynb)

<br>

## Exploratory Data Analysis
The distribution of aspects are similar between the train and test sets as shown by the following figure. As shown, ‘food’ comprises the largest proportion of all aspects, while ‘price’ and ‘ambience’, are least frequently discussed.

![figure3](/figures/fig3.png)

The following figure shows a word cloud presentation of the most frequent words included in each category.

![figure4](/figures/fig4.png)

<br>

## Analysis pipeline
[link to notebook](https://github.com/MahsaShokouhi/Aspect-Based_Sentiment_Analysis/blob/master/notebooks/pipeline.ipynb)

1. Aspect identification ([link to notebook](https://github.com/MahsaShokouhi/Aspect-Based_Sentiment_Analysis/blob/master/notebooks/aspect_modelling.ipynb))
There are two options for identifying aspects in a review: 1) supervised learning using labelled dataset, and 2) unsupervised topic modelling. The first approach can be implemented using a multi-label classification algorithm, since a review can contain multiple aspects. The second approach, can be implemented using any of the typical topic modelling methods, such as latent Dirichlet allocation, non-negative matrix factorization, etc. The extracted topics, can then be cross-referenced with aspects, for example using the cross-correlations between topics and aspects, to identify the aspects associated with them.
For this project, a voting ensemble was used to improve the prediction power. The ensemble of the following 4 models was used to identify the main topic(s) for each review:
* multi-label classification (supervised)
* latent Dirichlet allocation
* non-negative matrix factorization with Frobenius norm
* non-negative matrix factorization with Kullback-Leibler divergence

1.1. Supervised aspect detection using multi-label classification
Classifier chain algorithm, with logistic regression as the base model, was trained on the labelled data (train set). The model achieved average scores of 0.86, 0.79, 0.82 for precision, recall, and f1-score, respectively, on the test set.

1.2. Topic modeling
For topic modelling, two methods were examined: latent Dirichlet allocation (LDA), and non-negative matrix factorization (NMF). 
LDA, is a Bayesian method which finds topic using expectation maximization, where each topic can be represented by a group of words. 
NMF, on the other hand, is a dimensionality reduction method which decomposes the original the input data and transforms it obtain a smaller matrix, with fewer number of features. For topic modelling, this method is used to find a few topics from the original feature vector (vectorized text). The decomposed matrices can be used to reconstruct the original matrix and the loss is calculated based on the reconstruction error and can be evaluated using Frobenius norm or Kullback-Leibler divergence.

2. Sentiment analysis
Sentiment analysis was performed using NLYK library (VADER) to estimate the positive, negative, or neutral sentiment associated with a review.

3. Extracting review subject and descriptors ([link to notebook](https://github.com/MahsaShokouhi/Aspect-Based_Sentiment_Analysis/blob/master/notebooks/subject_descriptor_extraction.ipynb))
This final step of the pipeline provides more context about the review through information extraction. To understand the review more specifically, it’s useful to extract the subject of the review and its descriptors.

<br>

## Web Application

The analysis pipeline was used to develop a Flask app that takes a user review and performs aspect-based analysis. 

![figure5](/figures/fig5.png)

<br>

## Resources:
Dara: https://github.com/siat-nlp/MAMS-for-ABSA/tree/master/data/MAMS-ACSA/raw
