import org.apache.log4j.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.DecisionStump;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by iosifidis on 13.08.18.
 */
public class CrossFoldEvaluation {

    private final static Logger log = Logger.getLogger(CrossFoldEvaluation.class.getName());
    private static String protectedValueName;
    private static int protectedValueIndex;
    private static String targetClass;
    private static String otherClass;
    private static Classifier model;

    private FAE boost;

    public CrossFoldEvaluation(Classifier f,
                               int protectedValueIndex,
                               String protectedValueName,
                               String targetClass,
                               String otherClass,
                               Instances data,
                               double bound,
                               boolean useThreshold) throws Exception {

        boost = new FAE(f, protectedValueIndex, protectedValueName, targetClass, otherClass, bound, useThreshold);
        boost.buildClassifier(data);
    }

    public FAE getBoost() {
        return boost;
    }


    public static ArrayList<Double> FAEFaccuracy = new ArrayList<Double>();
    public static ArrayList<Double> FAEFauPRC = new ArrayList<Double>();
    public static ArrayList<Double> FAEFauROC = new ArrayList<Double>();
    public static ArrayList<Double> FAEFf1weighted = new ArrayList<Double>();
    public static ArrayList<Double> FAEFkappa = new ArrayList<Double>();
    public static ArrayList<Double> FAEFdiscList = new ArrayList<Double>();

    public static ArrayList<Double> SimpleModelDiscList = new ArrayList<Double>();
    public static ArrayList<Double> SimpleModelauROC = new ArrayList<Double>();

    public static ArrayList<Double> OBaccuracy = new ArrayList<Double>();
    public static ArrayList<Double> OBauPRC = new ArrayList<Double>();
    public static ArrayList<Double> OBauROC = new ArrayList<Double>();
    public static ArrayList<Double> OBf1weighted = new ArrayList<Double>();
    public static ArrayList<Double> OBdiscList = new ArrayList<Double>();
    public static ArrayList<Double> OBkappa = new ArrayList<Double>();


    public static ArrayList<Double> accuracythresholdBoosting = new ArrayList<Double>();
    public static ArrayList<Double> auPRCthresholdBoosting = new ArrayList<Double>();
    public static ArrayList<Double> auROCthresholdBoosting = new ArrayList<Double>();
    public static ArrayList<Double> f1weightedthresholdBoosting = new ArrayList<Double>();
    public static ArrayList<Double> kappaweightedthresholdBoosting = new ArrayList<Double>();

    public static ArrayList<Double> accuracyconfidenceBoosting = new ArrayList<Double>();
    public static ArrayList<Double> auPRCconfidenceBoosting = new ArrayList<Double>();
    public static ArrayList<Double> auROCconfidenceBoosting = new ArrayList<Double>();
    public static ArrayList<Double> f1weightedconfidenceBoosting = new ArrayList<Double>();
    public static ArrayList<Double> kappaweightedconfidenceBoosting = new ArrayList<Double>();



    public static double testForSE = 0;

    public static ArrayList<Double> discListthresholdBoosting = new ArrayList<Double>();
    public static ArrayList<Double> discListconfidenceBoosting = new ArrayList<Double>();

    public static void main(String[] argv) throws Exception {

        String modelString = argv[0];
        final String parameters = argv[1];

//        String modelString = "LR";
//        String parameters = "adult-gender";

        double bound = 0.0;
        BufferedReader reader = null;

        if (modelString.equals("NB")) {
            model = new NaiveBayes();
        } else if (modelString.equals("DS")) {
            model = new DecisionStump();
        } else if (modelString.equals("LR")) {
            Logistic xxx = new Logistic();
            if (parameters.equals("kdd")) {
                // big dataset use less iterations
                xxx.setMaxIts(5);
            }
            model = xxx;
        } else {
            System.exit(1);
        }

        if (parameters.equals("adult-gender")) {
//            reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff"));
            reader = new BufferedReader(new FileReader(argv[2]));
            protectedValueName = " Female";
            protectedValueIndex = 8;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (parameters.equals("adult-race")) {
//            reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff"));
            reader = new BufferedReader(new FileReader(argv[2]));
            protectedValueName = " Minorities";
            protectedValueIndex = 7;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (parameters.equals("dutch")) {
//            reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/dutch.arff"));
            reader = new BufferedReader(new FileReader(argv[2]));
            protectedValueName = "2"; // women
            protectedValueIndex = 0;
            targetClass = "2_1"; // high level
            otherClass = "5_4_9";
        } else if (parameters.equals("kdd")) {
//            reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/kdd.arff"));
            reader = new BufferedReader(new FileReader(argv[2]));
            protectedValueName = "Female";
            protectedValueIndex = 12;
            targetClass = "1";
            otherClass = "0";
        } else if (parameters.equals("german")) {
            protectedValueName = "female";
            protectedValueIndex = 20;
            targetClass = "bad";
            otherClass = "good";
        } else if (parameters.equals("propublica")) {
//            reader = new BufferedReader(new FileReader("/home/iosifidis/compass_zafar.arff"));
            reader = new BufferedReader(new FileReader(argv[2]));
            protectedValueName = "1";
            protectedValueIndex = 4;
            targetClass = "1";
            otherClass = "-1";
        } else {
            System.exit(1);
        }

        final Instances data = new Instances(reader);
        reader.close();
        int iterations = 20;
        provideStatistics(data);
        log.info("dataset = " + parameters);
        int k = 0;
        while (k < iterations) {
            k++;
            try {
                log.info("iteration = " + k);
                final Random rand = new Random((int) System.currentTimeMillis());   // create seeded number generator
                final Instances randData = new Instances(data);   // create copy of original data
                randData.randomize(rand);         // randomize data with number generator
                randData.setClassIndex(data.numAttributes() - 1);


                int trainSize = (int) Math.round(randData.numInstances() * 0.677);
                int testSize = randData.numInstances() - trainSize;
                Instances train = new Instances(randData, 0, trainSize);
                Instances test = new Instances(randData, trainSize, testSize);
                testForSE += testSize;

                Classifier newModel = AbstractClassifier.makeCopy(model);
                newModel.buildClassifier(train);
                double[] measures = EvaluateClassifier(newModel, train, test);
                double disc = equalOpportunityMeasurement(newModel, test, protectedValueIndex, protectedValueName, targetClass, otherClass);

                SimpleModelauROC.add(measures[2]);
                SimpleModelDiscList.add(disc);


                final CrossFoldEvaluation FAEF = new CrossFoldEvaluation(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, train, bound, true);
                final SDB confidenceBoosting = new SDB(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound);
                confidenceBoosting.setNumIterations(25);
                confidenceBoosting.buildClassifier(train);
                final SMT thresholdBoosting = new SMT(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound);
                thresholdBoosting.setNumIterations(25);
                thresholdBoosting.buildClassifier(train);

                measures = EvaluateClassifier(FAEF.getBoost(), train, test);
                disc = equalOpportunityMeasurement(FAEF.getBoost(), test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                FAEFaccuracy.add(measures[0]);
                FAEFauPRC.add(measures[1]);
                FAEFauROC.add(measures[2]);
                FAEFf1weighted.add(measures[3]);
                FAEFkappa.add(measures[4]);
                FAEFdiscList.add(disc);
//                double[] FBFpredictions = getBinaryPredictions(FAEF.getBoost(), test);

                // if EqOp is 0 then no special treatment is employed, this equals to OB model
                FAEF.getBoost().setEqOp(0.0);
                measures = EvaluateClassifier(FAEF.getBoost(), train, test);
                disc = equalOpportunityMeasurement(FAEF.getBoost(), test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                OBaccuracy.add(measures[0]);
                OBauPRC.add(measures[1]);
                OBauROC.add(measures[2]);
                OBf1weighted.add(measures[3]);
                OBkappa.add(measures[4]);
                OBdiscList.add(disc);
//                double[] Easypredictions = getBinaryPredictions(FAEF.getBoost(), test);

                measures = EvaluateClassifier(thresholdBoosting, train, test);
                disc = equalOpportunityMeasurement(thresholdBoosting, test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                accuracythresholdBoosting.add(measures[0]);
                auPRCthresholdBoosting.add(measures[1]);
                auROCthresholdBoosting.add(measures[2]);
                f1weightedthresholdBoosting.add(measures[3]);
                kappaweightedthresholdBoosting.add(measures[4]);
                discListthresholdBoosting.add(disc);
                double[] TBpredictions = getBinaryPredictions(thresholdBoosting, test);


                measures = EvaluateClassifier(confidenceBoosting, train, test);
                disc = equalOpportunityMeasurement(confidenceBoosting, test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                accuracyconfidenceBoosting.add(measures[0]);
                auPRCconfidenceBoosting.add(measures[1]);
                auROCconfidenceBoosting.add(measures[2]);
                f1weightedconfidenceBoosting.add(measures[3]);
                kappaweightedconfidenceBoosting.add(measures[4]);
                discListconfidenceBoosting.add(disc);
                double[] CBpredictions = getBinaryPredictions(confidenceBoosting, test);

             } catch (Exception e) {
                log.info("crushed, rerun iteration:");
                e.printStackTrace();
                k--;
            }

        }

        testForSE = testForSE / iterations;
        log.info("dataset = " + parameters);
//        calculateSD(FAEFaccuracy, "FAE Accuracy");
//        calculateSD(FAEFauPRC, "FAE Au-PRC");
        calculateSD(FAEFauROC, "FAE Au-ROC");
//        calculateSD(FAEFf1weighted, "FAE F1");
//        calculateSD(FAEFkappa, "FAE kappa");
        calculateSD(FAEFdiscList, "FAE disc");

//        calculateSD(OBaccuracy, "Easy Accuracy");
//        calculateSD(OBauPRC, "Easy Au-PRC");
        calculateSD(OBauROC, "OB Au-ROC");
//        calculateSD(OBf1weighted, "Easy F1");
//        calculateSD(OBkappa, "Easy kappa");
        calculateSD(OBdiscList, "OB disc");

//        calculateSD(accuracythresholdBoosting, "TB Accuracy");
//        calculateSD(auPRCthresholdBoosting, "TB Au-PRC");
        calculateSD(auROCthresholdBoosting, "SMT Au-ROC");
//        calculateSD(f1weightedthresholdBoosting, "TB F1");
//        calculateSD(kappaweightedthresholdBoosting, "TB kappa");
        calculateSD(discListthresholdBoosting, "SMT disc");

//        calculateSD(accuracyconfidenceBoosting, "CB Accuracy");
//        calculateSD(auPRCconfidenceBoosting, "CB Au-PRC");
        calculateSD(auROCconfidenceBoosting, "SDB Au-ROC");
//        calculateSD(f1weightedconfidenceBoosting, "CB F1");
//        calculateSD(kappaweightedconfidenceBoosting, "CB kappa");
        calculateSD(discListconfidenceBoosting, "SDB disc");



        calculateSD(SimpleModelauROC, "Model " +modelString + " Au-ROC");
        calculateSD(SimpleModelDiscList, "Model " + modelString + " disc");

    }


    private static void provideStatistics(Instances data) {
        data.setClassIndex(data.numAttributes() - 1);

        ArrayList<Instance> DG = new ArrayList<Instance>();
        ArrayList<Instance> DR = new ArrayList<Instance>();
        ArrayList<Instance> FG = new ArrayList<Instance>();
        ArrayList<Instance> FR = new ArrayList<Instance>();
        for (Instance instance : data) {
            if (instance.stringValue(protectedValueIndex).equals(protectedValueName)) {
                if (instance.stringValue(instance.classIndex()).equals(targetClass)) {

                    DG.add(instance);
                } else if (instance.stringValue(instance.classIndex()).equals(otherClass)) {
                    DR.add(instance);
                }
            } else {
                if (instance.stringValue(instance.classIndex()).equals(targetClass)) {
                    FG.add(instance);
                } else if (instance.stringValue(instance.classIndex()).equals(otherClass)) {
                    FR.add(instance);
                }
            }
        }

        log.info("total = " + data.size());
        log.info("positives = " + (DG.size() + FG.size()));
        log.info("negatives = " + (DR.size() + FR.size()));
        log.info("attribute number = " + data.numAttributes());
        log.info("DG = " + DG.size());
        log.info("FG = " + FG.size());
        log.info("numAttributes = " + data.numAttributes());

    }

    private static double[] getBinaryPredictions(Classifier clf, Instances test) throws Exception {
        double preds[] = new double[test.size()];
        int index = 0;
        for (Instance instance : test) {
            preds[index] = clf.classifyInstance(instance);
            index++;
        }

        return preds;
    }


    public static void calculateSD(ArrayList<Double> numArray, String measurement) {
        double sum = 0.0, standardDeviation = 0.0;

        for (double num : numArray) {
            sum += num;
        }

        double mean = sum / numArray.size();

        for (double num : numArray) {
            standardDeviation = standardDeviation + Math.pow(num - mean, 2) / numArray.size();
        }

        standardDeviation = Math.sqrt(standardDeviation);

        double standardError = standardDeviation / Math.sqrt(testForSE);
        log.info(measurement + " mean = " + mean + ", St.Error = " + standardError + ", st.Dev = " + standardDeviation);
    }


    public static double[] EvaluateClassifier(Classifier classifier, Instances training, Instances testing) throws Exception {

        final Evaluation eval = new Evaluation(training);
        eval.evaluateModel(classifier, testing);
        return new double[]{eval.pctCorrect(), eval.weightedAreaUnderPRC() * 100, eval.weightedAreaUnderROC() * 100, eval.weightedFMeasure() * 100, eval.kappa() * 100};
    }

    private static double equalOpportunityMeasurement(Classifier classifier,
                                                      Instances testing,
                                                      int protectedValueIndex,
                                                      String protectedValueName,
                                                      String targetClass,
                                                      String otherClass) throws Exception {

        Instances TestingPredictions = new Instances(testing);

        double tp_male = 0;
        double tn_male = 0;
        double tp_female = 0;
        double tn_female = 0;
        double fp_male = 0;
        double fn_male = 0;
        double fp_female = 0;
        double fn_female = 0;

        double correctlyClassified = 0.0;


        for (Instance ins : TestingPredictions) {
            double label = classifier.classifyInstance(ins);
            // WOMEN
            if (ins.stringValue(protectedValueIndex).equals(protectedValueName)) {
                // correctly classified
                if (label == ins.classValue()) {
                    correctlyClassified++;

                    // on target class (true negatives)
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_female++;

                    } else if (ins.stringValue(ins.classIndex()).equals(otherClass)) {
                        tn_female++;
                    }
                } else {
                    // error has been made on TN so it's FP
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        fn_female++;

                    } else if (ins.stringValue(ins.classIndex()).equals(otherClass)) {
                        // error has been made on TP so it's FN
                        fp_female++;

                    }
                }
            } else {
                // correctly classified
                if (label == ins.classValue()) {
                    correctlyClassified++;
                    // on target class (true negatives)
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_male++;

                    } else if (ins.stringValue(ins.classIndex()).equals(otherClass)) {
                        tn_male++;

                    }
                } else {
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        // error has been made on TP so it's FN

                        fn_male++;

                    } else if (ins.stringValue(ins.classIndex()).equals(otherClass)) {
                        // error has been made on TN so it's FP

                        fp_male++;

                    }
                }
            }
        }
        return 100 * ((tp_male) / (tp_male + fn_male) - (tp_female) / (tp_female + fn_female));
    }
}
