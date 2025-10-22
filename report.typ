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

= Choice of classification Algorithm

// Your final choice of classification algorithm, including the values of important
// parameters required by the algorithm. Communicate your approach in enough detail
// for someone else to be able to implement and deploy your spam filtering system.

= Pre processing

To pre-process our data, for different classifiers we tried the following different imputation strategies,

```py
if IMPUTATION_STRATEGY == 'mean':
    steps.append(('impute', SimpleImputer(missing_values=-1, strategy='mean')))
elif IMPUTATION_STRATEGY == 'median':
    steps.append(('impute', SimpleImputer(missing_values=-1, strategy='median')))
elif IMPUTATION_STRATEGY == 'knn':
    steps.append(('impute', KNNImputer(missing_values=-1, n_neighbors=5)))
elif IMPUTATION_STRATEGY == 'iterative':
  steps.append(('impute', IterativeImputer(missing_values=-1, random_state=1)))
```

And, for the chosen classifier that we ended up going with, random forests, we found that median imputation resulted in the most accurate classification rates.

To remove features, we tried using both a random forest and removing the lowest gini impurity classes, and also logistic regression with L1 regularization. It turned out that L1 regularization yielded higher accuracy, so we went with L1 regularization.


// Any pre-processing, such as exploratory data analysis, normalization, feature
// selection, etc.  that you performed, and how it impacted your results.

// How you selected and tested your algorithm and what other algorithms you
// compared against. Explain why you chose an algorithm and justify your decision!
// It is certainly OK to blindly test many algorithms, but you will likely find it
// a better use of your time to be selective based on the specifics of this data
// set and application.

// Recommendations on how to evaluate the effectiveness of your algorithm if it
// were to be deployed as a personalized spam filter for a user. What might be a
// good choice of metric, and what are the implications on the classifier? How
// might you solicit feedback from users to evaluate and improve your spam filter?