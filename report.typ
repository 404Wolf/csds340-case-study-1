#set document(title: "Title", author: "Author")

#set page(
  paper: "us-letter",
  margin: 1in,
  columns: 2
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

We also analyzed the Jaccard similarity of the dropped classes themselves, and found that in general about 60-70% of the classes that were dropped by either method were the same. The two methods did have their own classes that they would tend to drop, especially at increasing classes-dropped percentages, that were distinct from each other.

Ultimately, we decided only to go with l1 regularization for dropping features, because we felt we would already encapsulate low importance values in our actual classifier -- a decision tree classifier.

We found it interesting that, for logistic regression itself, we saw dramatically successful feature reduction with minimal loss to AUC. Performing l1 regularization on our data with $C=20$ removed *12* features while retaining *99.2%* of our AUC (in what seems to be a well-distributed decrease of AUC), for a simple logistic regression classifier.

We knew that some words should have no influence on the final output. Words like "the," "and," and the like should not really have much influence on whether it is spam or not. To try to eliminate "useless" words form consideration we used l1 regularization with our logistic regression model was something we decided to extract for our final classifier. This would reduce our risk of overfitting and also help remove redundant or useless features. 

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

This was not useful, so we proceeded to drop features using a random forest.

We assumed that it was likely that words in the email would co-occur because of english semantics, like if "free" were in the email, it would be more likely "money" would also show up. Getting a baseline for how independent our features were would be useful for choosing a final approach, so we proceeded by simply doing an initial comparison of a naïve bayes and logistic regression, and were surprised by the results.

#figure(
  image("images/bayes_vs_logistic.png"),
  caption: [On our data, naïve bayes seemed to perform marginally _better_ than logistic regression.],
)

Logistic regression performing _worse_ than naïve bayes was really surprising. This result led us to believe that the various attributes were more independent than we though (since one would indeed expect naïve bayes to perform better, or even optimally, if labels were more independent). The limitation of naïve bayes, though, and the only way we thought we'd be able to improve, is by dropping an assumption of our data having a linear decision boundary.

We explored two different general strategies to allow for nonlinear boundaries. First, training classifiers specifically designed to discern nonlinear boundaries, and second, preprocessing techniques to augment our data in ways that would induce a nonlinear decision boundary.

// Recommendations on how to evaluate the effectiveness of your algorithm if it
// were to be deployed as a personalized spam filter for a user. What might be a
// good choice of metric, and what are the implications on the classifier? How
// might you solicit feedback from users to evaluate and improve your spam filter?
