# Unsupervised Learning
 
The two datasets wwere provided during a university lab assignment and they are split into two types: a balanced group of data and an unbalanced group of data. This means that each class had the same amount of instances in the first set of data, while in the second one, a class was much more present than the other(s).


Both methods had the same ML methods deployed, with the difference in evaluation metrics. The methods used were: Neural Networks (MLP), SVM, Decision Tree and Logistic Regression. While the balanced data was trained and evaluated with the standard methods (with parameters tuned for the task), the unbalanced data was trained by taking into account how unbalanced each class is, by using weights attributed to each class. The more present a class is, the less its weight in training. The unbalanced data is evaluated with balanced accuracy instead of the standard accuracy (which is used for the balanced data).


The test results for the balanced data (just accuracies present here in this table):


| ML Method | Accuracy (%) |
| --------- | -----------  |
| MLP | 77 |
| SVM  | 75 |
| Decision Tree  | 70 |
| Logistic Reg.  | 72 |


And for the unbalanced data: 


| ML Method | Balanced Accuracy (%) |
| --------- | -----------  |
| MLP | 80 |
| SVM  | 83 |
| Decision Tree  | 74 |
| Logistic Reg.  | 83 |