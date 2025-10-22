#set document(title: "Title", author: "Author")

#set page(
  paper: "us-letter",
  margin: 1in,
)

#set text(
  size: 12pt,
)

#align(center)[
  #text(size: 17pt, weight: "bold")[CSDS340 Case Study]

  *Wolf Mermelstein* and *Alessandro Mason* \
  Case Western Reserve University
]

#set heading(numbering: "1.")


= Feature Selection

// Your final choice of classification algorithm, including the values of important parameters
// required by the algorithm. Communicate your approach in enough detail for someone else
// to be able to implement and deploy your spam filtering system. ( 5 points)

To start, we split up the data so that we could iterate safely while preserving the ability to judge our results. We wrote a simple script to split the data into three buckets: *initial test data ($20%$)*, *final test data ($40%$)*, and *actual training data ($40%$)*.

```py
data = np.loadtxt("spamTrain1.csv", delimiter=",")

indices = np.random.permutation(len(data)); shuffled_data = data[indices]

total_rows = len(shuffled_data); final_test_size = int(0.20 * total_rows)
start_test_size = int(0.40 * total_rows) # Remaining goes to data

final_test = shuffled_data[:final_test_size]
start_test = shuffled_data[final_test_size:final_test_size + start_test_size]
train_data = shuffled_data[final_test_size + start_test_size:]
```

We did this so that we could iterate with test data without our own observations influencing our actual expected test accuracy. As we experimented, we used the "initial test data."

To build intuition for the data that we were working with, we began by doing PCA analysis and creating some visual plots. To handle missing values we started with simple median interpolation.

#figure(
  image("images/pca_analysis.png"),
  caption: [PCA analysis. We found no useful associations in the reduced dimensionality data. The components did not add disparate amounts of variance, and overall variance seems like a useless separating characteristic. Note we ran this on all of the data, since the goal here was to build intuition as a human observer and not proceed with the reduced data.],
)

This was not useful, so we proceeded to drop features using a random forest.

We assumed that it was likely that words in the email would co-occur because of english semantics, like if "free" were in the email, it would be more likely "money" would also show up. Getting a baseline for how independent our features were would be useful for choosing a final approach, so we proceeded by simply doing an initial comparison of a naïve bayes and logistic regression, and were surprised by the results.

#figure(
  image("images/bayes_vs_logistic.png"),
  caption: [On our data, naïve bayes seemed to perform marginally _better_ than logistic regression.],
)

Logistic regression performing _worse_ than naïve bayes was really surprising. This result led us to believe that the various attributes were more independent than we though (since one would indeed expect naïve bayes to perform better, or even optimally, if labels were more independent). The limitation of naïve bayes, though, and the only way we thought we'd be able to improve, is by dropping an assumption of our data having a linear decision boundary.

We explored two different general strategies to allow for nonlinear boundaries. First, training classifiers specifically designed to discern nonlinear boundaries, and second, preprocessing techniques to augment our data in ways that would induce a nonlinear decision boundary.

We focused primarily on this second idea of trying to get a logistic regression classifier to work by forcing our data to be linearly separable enough with pre-processing.

We knew that some words should have no influence on the final output. Words like "the," "and," and the like should not really have much influence on whether it is spam or not. To try to eliminate "useless" words form consideration we used l1 regularization with our logistic regression model.

#figure(
  image("images/logistic_l1_comparison.png"),
  caption: [Performing l1 regularization on our data with $C=20$ removed *12* features while retaining *99.2%* of our AUC (in what seems to be a well-distributed decrease of AUC)],
)

Ultimately we think *l1 regularization* worked really well. We were able to eliminate many features while retaining a high AUC. We wanted to be disciplined in approach to choose a $C$, so we plotted AUCs and retained feature counts for different $C$ values.

At the same time, we tried using random forest feature selection. We tried this as well since we thought it'd be possible (but we were not certain) for there to be "garbage" features that have nonlinear correlation. We also plotted a simple random forest classifier with different numbers of retained features to get a sense of this.

#figure(
  image("images/l1_feature_selection.png"),
  caption: [Performing l1 regularization for different $C$ values, and observing its affect on the AUC. Then, also seeing how _number of retained features_ benefits the AUC for random forests.],
)

Based on these experiments it looked like there was general consensus between the two that half of the features were most useful. We ended up choosing $C=16$, which eliminates a bit less than half the features, but does not really compromise accuracy very much.

= Pre processing

// Any pre-processing, such as exploratory data analysis, normalization, feature
// selection, etc.  that you performed, and how it impacted your results. (10
// points)

In order to actually "beat" naïve bayes with a linear classifier though, we would need to augment our data, since as-is it did not seem to be the case that our data was properly linearly separable, which was what we suspected was holding back our logistic regression classifier.

We researched preprocessing techniques for text frequency datasets, and also came across TF-IDF. The idea behind TF-IDF is that you can determine how "important" a word is to some specific document based on how common it is in that specific document as compared to all other documents in the dataset. You reweight your dataset using it, and then when you train, say, a regression model, you give precedence to the more "extreme" features. It seemed perfect for this sort of use case, so we applied it in as a preprocessing step.

#figure(
  image("images/logistic_tfidf_comparison.png"),
  caption: [Using TF-IDF before applying a logistic classifier. We were able to outperform naïve bayes.]
)

= Algorithm Selection

// How you selected and tested your algorithm and what other algorithms you
// compared against. Explain why you chose an algorithm and justify your decision!
// It is certainly OK to blindly test many algorithms, but you will likely find it
// a better use of your time to be selective based on the specifics of this data
// set and application. (10 points)

Ultimately, we settled on a core of a logistic regression model, with extensive preprocessing.

#figure(
  image("images/model.png"),
  caption: [The general algorithm, including preprocessing and classifiers we used for our final lgorithm],
)

= Personal Spam Filter

// Recommendations on how to evaluate the effectiveness of your algorithm if it were to be
// deployed as a personalized spam filter for a user. What might be a good choice of metric,
// and what are the implications on the classifier? How might you solicit feedback from users
// to evaluate and improve your spam filter? (10 points)

We will note that we did not explore it for the bulk of our experimentation, but for a case like this it may make the most sense to use a k-nearest-neighbor classifier.