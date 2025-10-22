#set document(title: "Title", author: "Author")

#set page(
  paper: "us-letter",
  margin: 1in,
  columns: 2,
)

#set text(
  size: 12pt,
)

#show link: set text(fill: blue)

#align(center)[
  #text(size: 17pt, weight: "bold")[CSDS340 Case Study]

  *Wolf Mermelstein* and *Alessandro Mason* \
  Case Western Reserve University
]

#set heading(numbering: "1.")

= Our chosen algorithm

// Your final choice of classification algorithm, including the values of important
// parameters required by the algorithm. Communicate your approach in enough detail
// for someone else to be able to implement and deploy your spam filtering system.

= Pre processing

// Any pre-processing, such as exploratory data analysis, normalization, feature
// selection, etc.  that you performed, and how it impacted your results.

To pre-process our data, we sought to address a few issues:
+ The data, as given, for each email, often contained many frequencies that were missing.
+ The data was overly complex. There were many features that did not have much importance, and generally was adding unnecessary complexity. Additionally, we were told that some classes in the dataset had literally no relevance.

To make up for 1., we experimented by testing out different imputation strategies, including:

- Mean imputation
- Median imputation
- Knn imputation
- Iterative imputation

Using `scikit-learn`'s `SimpleImputer`, `KNNImputer`, and `IterativeImputer`.

In all cases, for our chosen classifier we found that median imputation resulted in the most accurate classification rates.

For 2., to remove features, we tried using both a random forest and removing the lowest gini impurity classes, and also logistic regression with L1 regularization. It turned out that L1 regularization yielded higher accuracy, so we went with L1 regularization.

To find the best choice of $C$ for logistic regression (the hyperparameter that controls the inverse of regularization strength, where higher $C$ means less regularization), we analyzed the effect of different $C$ values and observed corresponding AUCs (seen in @fig:auc-by-c).

#figure(
  image("images/c_analysis_plot.png"),
  caption: [Analyzing the AUC for different choices of $C$, paying attention to the corresponding number of features that get dropped at each level],
) <fig:auc-by-c>

We also considered dropping features by pre-processing with a round of random forest (feature importance based feature selection) before running our classifier. We looked at the effect of dropping different numbers of features on the AUC (seen in @fig:auc-by-features-dropped).

#figure(
  image("images/trees_drop_analysis.png"),
  caption: [Analyzing the AUC for when we drop different lowest-importance classes based on random forest classification],
) <fig:auc-by-features-dropped>

#figure(
  image("images/feature_importance_analysis.png"),
  caption: [The importance of various features. The drop off is mostly linear with two outliers. For the most part, features are decreasingly "important," with no exceptional strata for importance],
)

We also analyzed the Jaccard similarity of the dropped classes themselves, and found that in general about 60-70% of the classes that were dropped by either method were the same (across different C values and \# important features dropped). The two methods did have their own classes that they would tend to drop, especially at increasing classes-dropped percentages, that were distinct from each other.

Ultimately, we decided only to go with l1 regularization for dropping features, because we felt we would already encapsulate low importance values in our actual classifier -- a decision tree classifier.

We found it interesting that, for logistic regression itself, we saw dramatically successful feature reduction with minimal loss to AUC. Performing l1 regularization on our data with $C=20$ removed *12* features while retaining *99.2%* of our AUC (in what seems to be a well-distributed decrease of AUC), for a simple logistic regression classifier.

We knew that some words should have no influence on the final output. Words like "the," "and," and the like should not really have much influence on whether it is spam or not. To try to eliminate "useless" words form consideration we used l1 regularization with our logistic regression model was something we decided to extract for our final classifier. This would reduce our risk of overfitting and also help remove redundant or useless features.

In addition to these approaches, we also found in various tangential research often made use of _TF-IDF_, where you reweight your data such that word frequencies in a given document get amplified if the given word for that document is relatively rare in the dataset and common in that given document @janez_martino. We used scicpy's implementation of this to try improving logistic regression -- since, we suspected, it would result in a _more_ linearly separable dataset if we were moving datapoints to the "correct"/more extreme direction.

#figure(
  image("images/logistic_tfidf_comparison.png"),
  caption: [Logistic regression with and without TF-IDF, compared to Bernoulli naïve bayes with and without TF-IDF. Note that Bernoulli naïve bayes does experience any benefit from TF-IDF since features are only compared within themselves.]
) <fig:tfidf-comparision>

As seen in @fig:tfidf-comparision, we were able to achieve a about 3% higher AUC using this approach with logistic regression. We tried also pre-processing the data in this way for random forest analysis, but it didn't have any significant improvement.

= Choosing our Algorithm

// How you selected and tested your algorithm and what other algorithms you
// compared against. Explain why you chose an algorithm and justify your decision!
// It is certainly OK to blindly test many algorithms, but you will likely find it
// a better use of your time to be selective based on the specifics of this data
// set and application.

To build intuition for the data that we were working with, we began by doing PCA analysis and creating some visual plots. To handle missing values we started with simple median interpolation.

#figure(
  image("images/pca_analysis.png"),
  caption: [PCA analysis. We found no useful associations in the reduced dimensionality data. The components did not add disparate amounts of variance, and overall variance seems like a useless separating characteristic. Note we ran this on all of the data, since the goal here was to build intuition as a human observer and not proceed with the reduced data.],
)

We assumed that it was likely that words in the email would co-occur because of english semantics, like if "free" were in the email, it would be more likely "money" would also show up. Getting a baseline for how independent our features were would be useful for choosing a final approach, so we proceeded by simply doing an initial comparison of a naïve bayes and logistic regression, and were surprised by the results.

We performed pre-processing to drop features, thinking that we may be able to improve results by focusing the classifier on features that have dependant relationships that naïve bayes could not account for well. However, even after dropping features, we found that logistic regression underperformed.

#figure(
  image("images/bayes_vs_logistic.png"),
  caption: [On our data, naïve bayes seemed to perform marginally _better_ than logistic regression.],
)

This result led us to believe that the various attributes were more independent than we though (since one would indeed expect naïve bayes to perform better, or even optimally, if labels were more independent). The limitation of naïve bayes, though, and the only way we thought we'd be able to improve, is by dropping an assumption of our data having a linear decision boundary.

We explored two different general strategies to allow for nonlinear boundaries. First, training classifiers specifically designed to discern nonlinear boundaries, and second, pre-processing techniques to augment our data in ways that would induce a nonlinear decision boundary.

// Recommendations on how to evaluate the effectiveness of your algorithm if it
// were to be deployed as a personalized spam filter for a user. What might be a
// good choice of metric, and what are the implications on the classifier? How
// might you solicit feedback from users to evaluate and improve your spam filter?

= Personal Spam Filter

== New metrics

// Add the new metrics and why we think they are good new metrics

The current metrics that we are using, AUC, and TPR \@ FPR=1%, are probably still roughly reasonable for this new situation given the constraints of the problem. We still would like to classify emails correctly and be very careful to not label emails that are not spam as spam.

Since we are letting users have more control over their spam filter though, we may, however, want to work in the ability for them to actually configure their own FNR that they are comfortable with.

== Soliciting feedback

Assuming the same words are the words that a user would want to use to flag emails as spam, we could proceed by just appending word frequency rows to our dataset, and retraining every time that the user would "flag" a given email that our predictor did not previously flag. This is the natural approach to choose, but it is unideal because, at least for our process, training with random forests is relatively expensive, and would not scale well.  We could, instead, compromise and choose to use a worse classifier, like _logistic regression_ with _TF-IDF_ as we explored earlier, which could be relatively cheap to train, or could choose a classifier that does not require training at all, like a _K Nearest Neighbor_ classifier. The issue with using a nearest neighbor classifier is that it would be really expensive to categorize a given email as spam, and we are always classifying emails as spam or not more often then users are flagging emails.

Ultimately, however, we just do not feel that using word frequencies for only the existing frequencies is sufficient to create a good custom spam classifier.

To solicit feedback, we would likely want to build on prior art, and try to gather as much information as possible when a user "marks as spam" an email. Since it is a personalized filter, it is possible that they flag an email as spam not because _any_ of the frequencies. This introduces an immediate issue, since by just appending a document to our dataset and retraining would probably not work, since here may be unique words that tipped them off.

To try to figure out what "tipped them off," we can look for important words with NLP methods. We are somewhat limited though, since we will not have data on the frequencies of new words. Because of this, to find words that probably tipped them off, we would have to use metrics like pointwise mutual information (PMI). However, we may have "important" words in the email that have nothing to do with whether it is spam or why they marked it as spam. If we keep all the previous documents, we would have dramatically higher storage requirements, but would be able to use more powerful methods like TF-IDF.

When we identify new "important" words that were not in our original data, we would end up adding new columns and median-imputing a median of a very small number of points, which would then be implying that that the in-feature variance would be very low, and it would on its own be a useless additional feature until we receive sufficient additional documents.

== New Algorithms

Our current random forest with pre-processing might work for a customized spam classifier, but the fundamental issue is that we would have to retrain it every time.

=== Embeddings w/K-Nearest-Neighbor

The most effective approach would be to start from scratch, since there are in general many fundamental disadvantages to our current word frequency approach.

One unique idea that we explored is using vector embeddings instead of simple word frequencies. Vector embeddings convert each email into a high-dimensional vector that encodes semantic information. This allows us to compare distances between vectors and capture complex semantic relationships in the email content.

The advantage of this approach is that semantically similar emails will be close together in the embedding space, regardless of whether they share exact word frequencies. When a user flags an email as spam, we vectorize it and add it to our corpus of labeled examples. We can then use a simple K-Nearest-Neighbor classifier to determine if new emails are similar to previously flagged spam or legitimate emails.

This method naturally adapts to personalized spam definitions without requiring expensive retraining, and it captures semantic meaning that word frequencies alone cannot represent.

We explored what the implementation of this would look like, and found we could get something working with #link("https://huggingface.co/sentence-transformer", "sentence-transformers"), and an open embedding source embedding model (we found "all-MiniLM-L6-v2" works well and is fast). We produced a working example that shows what this would look like #link("https://gist.github.com/404Wolf/3685d01d1224aa86fb6f62621fcbd22a", "here").

#place(
  top,
  float: true,
  scope: "parent",
  figure(
    ```py
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def email_to_vector(email_text: str):
        return model.encode(email_text, normalize_embeddings=True)

    def k_nearest_majority_vote(email_vec, spam_vectors, ham_vectors, k = 5):
        all_vectors = np.vstack([spam_vectors, ham_vectors])
        labels = np.array(["spam"] * len(spam_vectors) + ["ham"] * len(ham_vectors))

        sims = cosine_similarity(email_vec.reshape(1, -1), all_vectors).ravel()
        top_k_idx = np.argsort(sims)[-k:]; top_k_labels = labels[top_k_idx]

        pred = max(set(top_k_labels), key=list(top_k_labels).count); return pred
    ```,
    caption: align(left, [
      Email spam classifier using k-NN algorithm. The `email_to_vector` function converts email text into a 384-dimensional semantic embedding using a pretrained transformer model. The `k_nearest_majority_vote` function classifies an email by

      + Combining all known spam and ham embeddings
      + Computing cosine similarity between the input email and all training examples
      + Selecting the k=5 most similar emails
      + Predicting the class by majority vote among these neighbors
    ]),
  ),
)


// To solve this, we will take advantage of


#bibliography("sources.bib")