# Fairness-Aware-Ensemble-Framework

Fairness-Aware-Ensemble-Framework (FAEF) is a framework which deals with discrimination when population imbalance exists in the dataset. It can be employed with any classifier (as weak learner) combined with Boosting.

In addition, FAEF can be altered to accomodate fairness functions which take into consideration true and predicted labels.

In our setup we employ equal opportunity as fairness function that checks the difference of true positive ratios among two given populations. Given a threshold parameter $epsilon$, the user can define the level of discrimination which is allowed by the model. A strict and fair model must have epsilon = 0. 

