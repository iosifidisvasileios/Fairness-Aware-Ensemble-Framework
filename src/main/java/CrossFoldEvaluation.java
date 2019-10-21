import org.apache.log4j.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
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
    private static String parameters;

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
        boost.setParallelization(true);
        if (parameters.equals("kdd")){
            boost.setParallelization(false);
        }
        boost.buildClassifier(data);
    }

    public FAE getBoost() {
        return boost;
    }

    public static ArrayList<Double> FAEF_EM_discList = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_EM_balacc = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_EM_protected_tp = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_EM_non_protected_tp = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_EM_protected_tn = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_EM_non_protected_tn = new ArrayList<Double>();

    public static ArrayList<Double> FAEF_KNN_discList = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_KNN_balacc = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_KNN_protected_tp = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_KNN_non_protected_tp = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_KNN_protected_tn = new ArrayList<Double>();
    public static ArrayList<Double> FAEF_KNN_non_protected_tn = new ArrayList<Double>();


    public static ArrayList<Double> OB_EM_discList = new ArrayList<Double>();
    public static ArrayList<Double> OB_EM_balacc = new ArrayList<Double>();
    public static ArrayList<Double> OB_EM_protected_tp = new ArrayList<Double>();
    public static ArrayList<Double> OB_EM_non_protected_tp = new ArrayList<Double>();
    public static ArrayList<Double> OB_EM_protected_tn = new ArrayList<Double>();
    public static ArrayList<Double> OB_EM_non_protected_tn = new ArrayList<Double>();


    public static ArrayList<Double> OB_KNN_discList = new ArrayList<Double>();
    public static ArrayList<Double> OB_KNN_balacc = new ArrayList<Double>();
    public static ArrayList<Double> OB_KNN_protected_tp = new ArrayList<Double>();
    public static ArrayList<Double> OB_KNN_non_protected_tp = new ArrayList<Double>();
    public static ArrayList<Double> OB_KNN_protected_tn = new ArrayList<Double>();
    public static ArrayList<Double> OB_KNN_non_protected_tn = new ArrayList<Double>();




    public static void main(String[] argv) throws Exception {

        String modelString = argv[0];
        parameters = argv[1];
        int iterations = 10;
//        String modelString = "DS";
//        parameters = "bank";

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
                xxx.setMaxIts(1);
            }
            model = xxx;
        } else {
            System.exit(1);
        }

        if (parameters.equals("adult-gender")) {
            reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff"));
//            reader = new BufferedReader(new FileReader(argv[2]));
            protectedValueName = " Female";
            protectedValueIndex = 7;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (parameters.equals("kdd")) {
            reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/kdd.arff"));
//            reader = new BufferedReader(new FileReader(argv[2]));
            iterations = 3;
            protectedValueName = "Female";
            protectedValueIndex = 12;
            targetClass = "1";
            otherClass = "0";
        } else if (parameters.equals("propublica")) {
            reader = new BufferedReader(new FileReader("/home/iosifidis/compass_zafar.arff"));
//            reader = new BufferedReader(new FileReader(argv[2]));
            protectedValueName = "1";
            protectedValueIndex = 4;
            targetClass = "1";
            otherClass = "-1";
        }  else if (parameters.equals("bank")) {
            reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/bank-full.arff"));
            protectedValueName = "single";
            protectedValueIndex = 2;
            targetClass = "yes";
            otherClass = "no";
        } else {
            System.exit(1);
        }

        final Instances data = new Instances(reader);
        reader.close();


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

                AdaBoostM1 adaBoostM1 = new AdaBoostM1();
                adaBoostM1.setClassifier(AbstractClassifier.makeCopy(model));
                adaBoostM1.setNumIterations(25);
                adaBoostM1.buildClassifier(train);






                FAE FAEF_EM = new FAE(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound, true);
                FAEF_EM.setParallelization(true);
                FAEF_EM.setMaxClusterIteration(10);
                if (parameters.equals("kdd")){
                    FAEF_EM.setParallelization(false);
                }
                FAEF_EM.setDECIDE_CLUSTER_METHOD(0);
                FAEF_EM.buildClassifier(train);

                double[] measures = EvaluateClassifier(FAEF_EM, train, test);
                double[] disc = equalOpportunityMeasurement(FAEF_EM, test, protectedValueIndex, protectedValueName, targetClass, otherClass);


                FAEF_EM_balacc.add(measures[5]);
                FAEF_EM_discList.add(disc[4]);
                FAEF_EM_protected_tp.add(disc[0]);
                FAEF_EM_non_protected_tp.add(disc[1]);
                FAEF_EM_protected_tn.add(disc[2]);
                FAEF_EM_non_protected_tn.add(disc[3]);

                // if EqOp is 0 then no special treatment is employed, this equals to OB model
                FAEF_EM.setEqOp(0.0);
                measures = EvaluateClassifier(FAEF_EM, train, test);
                log.info("OB EM");

                disc = equalOpportunityMeasurement(FAEF_EM, test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                OB_EM_balacc.add(measures[5]);
                OB_EM_discList.add(disc[4]);
                OB_EM_protected_tp.add(disc[0]);
                OB_EM_non_protected_tp.add(disc[1]);
                OB_EM_protected_tn.add(disc[2]);
                OB_EM_non_protected_tn.add(disc[3]);

                //////////////////////////////////////////////////////

                FAE FAEF_KNN = new FAE(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound, true);
                FAEF_KNN.setParallelization(true);
                FAEF_KNN.setMaxClusterIteration(10);

                if (parameters.equals("kdd")){
                    FAEF_KNN.setParallelization(false);
                }
                FAEF_KNN.setDECIDE_CLUSTER_METHOD(1);
                FAEF_KNN.buildClassifier(train);

                measures = EvaluateClassifier(FAEF_KNN, train, test);
                disc = equalOpportunityMeasurement(FAEF_KNN, test, protectedValueIndex, protectedValueName, targetClass, otherClass);


                FAEF_KNN_balacc.add(measures[5]);
                FAEF_KNN_discList.add(disc[4]);
                FAEF_KNN_protected_tp.add(disc[0]);
                FAEF_KNN_non_protected_tp.add(disc[1]);
                FAEF_KNN_protected_tn.add(disc[2]);
                FAEF_KNN_non_protected_tn.add(disc[3]);

                // if EqOp is 0 then no special treatment is employed, this equals to OB model
                FAEF_KNN.setEqOp(0.0);
                measures = EvaluateClassifier(FAEF_KNN, train, test);
                log.info("OB KNN");

                disc = equalOpportunityMeasurement(FAEF_KNN, test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                OB_KNN_balacc.add(measures[5]);
                OB_KNN_discList.add(disc[4]);
                OB_KNN_protected_tp.add(disc[0]);
                OB_KNN_non_protected_tp.add(disc[1]);
                OB_KNN_protected_tn.add(disc[2]);
                OB_KNN_non_protected_tn.add(disc[3]);



            } catch (Exception e) {
                log.info("crushed, rerun iteration:");
                e.printStackTrace();
                k--;
            }

        }

        log.info("dataset = " + parameters);
        log.info("modelString = " + modelString);

        log.info("**************************************************");
        calculateSD(FAEF_EM_balacc, "FAEF EM balacc");
        calculateSD(FAEF_EM_discList, "FAEF EM disc");
        calculateSD(FAEF_EM_protected_tp, "FAEF EM Prot. TP");
        calculateSD(FAEF_EM_non_protected_tp, "FAEF EM Non-Prot. TP");
        calculateSD(FAEF_EM_protected_tn, "FAEF EM Prot. TN");
        calculateSD(FAEF_EM_non_protected_tn, "FAEF EM Non-Prot. TN");
        log.info("**************************************************");


        calculateSD(OB_EM_balacc, "OB EM balacc");
        calculateSD(OB_EM_discList, "OB EM disc");
        calculateSD(OB_EM_protected_tp, "OB EM Prot. TP");
        calculateSD(OB_EM_non_protected_tp, "OB EM Non-Prot. TP");
        calculateSD(OB_EM_protected_tn, "OB EM Prot. TN");
        calculateSD(OB_EM_non_protected_tn, "OB EM Non-Prot. TN");
        log.info("**************************************************");

        calculateSD(FAEF_KNN_balacc, "FAEF KNN balacc");
        calculateSD(FAEF_KNN_discList, "FAEF KNN disc");
        calculateSD(FAEF_KNN_protected_tp, "FAEF KNN Prot. TP");
        calculateSD(FAEF_KNN_non_protected_tp, "FAEF KNN Non-Prot. TP");
        calculateSD(FAEF_KNN_protected_tn, "FAEF KNN Prot. TN");
        calculateSD(FAEF_KNN_non_protected_tn, "FAEF KNN Non-Prot. TN");
        log.info("**************************************************");


        calculateSD(OB_KNN_balacc, "OB KNN balacc");
        calculateSD(OB_KNN_discList, "OB KNN disc");
        calculateSD(OB_KNN_protected_tp, "OB KNN Prot. TP");
        calculateSD(OB_KNN_non_protected_tp, "OB KNN Non-Prot. TP");
        calculateSD(OB_KNN_protected_tn, "OB KNN Prot. TN");
        calculateSD(OB_KNN_non_protected_tn, "OB KNN Non-Prot. TN");
        log.info("**************************************************");

    }

    private static int provideStatistics(Instances data) {
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
        int maximum = -1;

        if (DR.size() > maximum)
            maximum = DR.size();
        if (FR.size() > maximum)
            maximum = FR.size();
        if (FG.size() > maximum)
            maximum = FG.size();

        int bags = (int) (1 / ((double) DG.size() / (double) maximum)) + 1;
        return bags;
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

//        double standardError = standardDeviation / Math.sqrt(testForSE);
        log.info(measurement + " mean = " + mean );//+ ", St.Error = " + standardError + ", st.Dev = " + standardDeviation);
    }



    public static double[] EvaluateClassifier(Classifier classifier, Instances training, Instances testing) throws Exception {

        final Evaluation eval = new Evaluation(training);
        eval.evaluateModel(classifier, testing);
        double balanced_acc =100* (eval.trueNegativeRate(1) + eval.truePositiveRate(1)) / 2;
        return new double[]{eval.pctCorrect(), eval.weightedAreaUnderPRC() * 100, eval.weightedAreaUnderROC() * 100, eval.weightedFMeasure() * 100, eval.kappa() * 100, balanced_acc };
    }

    private static double [] equalOpportunityMeasurement(Classifier classifier,
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


        double [] output = new double[5];
        output[0] = 100*(tp_female) / (tp_female + fn_female);
        output[1] =  100*(tp_male) / (tp_male + fn_male);
        output[2] = 100*(tn_female) / (tn_female + fp_female);
        output[3] = 100*(tn_male) / (tn_male + fp_male);
        output[4] = 100 * ((tp_male) / (tp_male + fn_male) - (tp_female) / (tp_female + fn_female));

        return output;
    }
}
