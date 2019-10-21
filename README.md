# Fairness-Aware-Ensemble Framework

Fairness-Aware-Ensemble (FAE) Framework is a framework which deals with discrimination when population imbalance exists in the dataset. It can be employed with any classifier (as weak learner) combined with Boosting.

FAE is a combination of pre- and post-processing fairness enhancing interventions which aim to eliminate algorithmic discrimination towards specific groups or individuals. In the pre-processing phase a bagging schema is employed in which balanced and well-represented samples of the data occupy the bags. During the training phase these fair bags are provided to AdaBoost classifiers. After the training phase, a fairness function is calculated based on the majority vote of the ensemble to determine if bias exists. If bias exists, then FAEF executes a post-processing step in which a boundary is calculated based on the training set that will minimize discrimination.  

In our setup we employ equal opportunity as fairness function that checks the difference of true positive ratios among two given populations. Given a threshold parameter epsilon, the user can define the level of discrimination which is allowed by the model. A fair model must have epsilon = 0, however this may have a serious impact on the overall performance. 

In addition, FAE can be altered to accomodate fairness functions which take into consideration true and predicted labels. 

This repo contains the source code of FAE written in java. FAE file contains our framework which can be compared with other methods CrossFoldEvaluation file. The project is under maven support.

So far the code supports only weka instances thus arff files. Also, default parameters such as favored and deprived popoulations and target class have been initialised in CrossFoldEvaluation class for specific datasets such as adult-census, adult-census-kdd, dutch-census and propublica.  
