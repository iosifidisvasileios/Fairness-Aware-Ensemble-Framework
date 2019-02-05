import org.apache.log4j.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.DecisionStump;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;

/**
 * Created by iosifidis on 13.08.18.
 */
public class PredictAndOutputForSignifTest {

    private final static Logger log = Logger.getLogger(PredictAndOutputForSignifTest.class.getName());
    private static String protectedValueName;
    private static int protectedValueIndex;
    private static String targetClass;
    private static String otherClass;
    private static Classifier model;

//    public static double t10McNemarScoreCB = 0;
//    public static double t10McNemarScoreTB = 0;

    private FAE boost;

    public PredictAndOutputForSignifTest(Classifier f,
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


    public static void main(String[] argv) throws Exception {

        // get train and test from DM output for significant test
        String modelString = argv[0];
        final String parameters = argv[1];

//        String modelString = "LR";
//        String parameters = "dutch";

        double bound = 0.0;
        BufferedReader readerTrain = null;
        BufferedReader readerTest = null;

        if (modelString.equals("NB")) {
            model = new NaiveBayes();
        } else if (modelString.equals("DS")) {
            model = new DecisionStump();
        } else if (modelString.equals("LR")) {
            Logistic xxx = new Logistic();
            if (parameters.equals("kdd")) {
//             big dataset use less iterations
                xxx.setMaxIts(5);
            }
            model = xxx;
        } else {
            System.exit(1);
        }

        if (parameters.equals("adult-gender")) {
//            readerTrain = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff"));
            readerTrain = new BufferedReader(new FileReader(argv[2]));
            readerTest = new BufferedReader(new FileReader(argv[3]));
            protectedValueName = " Female";
            protectedValueIndex = 8;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (parameters.equals("adult-race")) {
//            readerTrain = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff"));
            readerTrain = new BufferedReader(new FileReader(argv[2]));
            readerTest = new BufferedReader(new FileReader(argv[3]));
            protectedValueName = " Minorities";
            protectedValueIndex = 7;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (parameters.equals("dutch")) {
//            readerTrain = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/dutch.arff"));
            readerTrain = new BufferedReader(new FileReader(argv[2]));
            readerTest = new BufferedReader(new FileReader(argv[3]));
            protectedValueName = "2"; // women
            protectedValueIndex = 0;
            targetClass = "2_1"; // high level
            otherClass = "5_4_9";
        } else if (parameters.equals("kdd")) {
//            readerTrain = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/kdd.arff"));
            readerTrain = new BufferedReader(new FileReader(argv[2]));
            readerTest = new BufferedReader(new FileReader(argv[3]));
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
//            readerTrain = new BufferedReader(new FileReader("/home/iosifidis/PycharmProjects/fair-classification/disparate_mistreatment/propublica_compas_data_demo/compass_training.arff"));
//            readerTest = new BufferedReader(new FileReader("/home/iosifidis/PycharmProjects/fair-classification/disparate_mistreatment/propublica_compas_data_demo/compass_testing.arff"));
            readerTrain = new BufferedReader(new FileReader(argv[2]));
            readerTest = new BufferedReader(new FileReader(argv[3]));
            protectedValueName = "1";
            protectedValueIndex = 4;
            targetClass = "1";
            otherClass = "-1";
        } else {
            System.exit(1);
        }

        final Instances train = new Instances(readerTrain);
        final Instances test = new Instances(readerTest);
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

/*
        final Random rand = new Random((int) System.currentTimeMillis());   // create seeded number generator
        final Instances randData = new Instances(readerTrain);   // create copy of original data
        randData.randomize(rand);         // randomize data with number generator
        randData.setClassIndex(randData.numAttributes() - 1);

        int trainSize = (int) Math.round(randData.numInstances() * 0.677);
        int testSize = randData.numInstances() - trainSize;

        Instances train = new Instances(randData, 0, trainSize);
        Instances test = new Instances(randData, trainSize, testSize);
*/

        final PredictAndOutputForSignifTest FAEF = new PredictAndOutputForSignifTest(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, train, bound, true);
        final SDB sdb = new SDB(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound);
        sdb.setNumIterations(25);
        sdb.buildClassifier(train);
        final SMT smt = new SMT(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound);
        smt.setNumIterations(25);
        smt.buildClassifier(train);

        double[] FAEFpredictions = getBinaryPredictions(FAEF.getBoost(), test);
        // if EqOp is 0 then no special treatment is employed, this equals to OB model
        FAEF.getBoost().setEqOp(0.0);
        double[] OBpredictions = getBinaryPredictions(FAEF.getBoost(), test);
        double[] SMTpredictions = getBinaryPredictions(smt, test);
        double[] SDBpredictions = getBinaryPredictions(sdb, test);
        model.buildClassifier(train);
        double[] Simplepredictions = getBinaryPredictions(model, test);

        outputPredictionsToFile(Simplepredictions, SMTpredictions, SDBpredictions,OBpredictions, FAEFpredictions, parameters, modelString);
    }


    private static void outputPredictionsToFile(double[] SimpleModel, double[] SMT, double[] SDB, double[] OB, double[] FAEF, String parameters, String modelString) throws IOException {
        FileWriter fstream = new FileWriter(parameters + "_" + modelString + ".tsv"); //true tells to append data.
        BufferedWriter out = new BufferedWriter(fstream);
        out.write("index\tDM\tSimpleModel\tSDB\tSMT\tOB\tFAEF\n");
        for (int i = 0; i < FAEF.length; i++) {
            out.write(i + "\t\t" + SimpleModel[i] + "\t" + SDB[i] + "\t" + SMT[i] + "\t" + OB[i] + "\t" + FAEF[i] + "\n");
        }
        out.close();
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
/*

    private static void calculateMcNemarSignificantScore(double[] ComparingPredictions,
                                                         double[] t10predictions,
                                                         String option) {

        double[] score10 = calculateVectorDifference(ComparingPredictions, t10predictions);

        if (option.equals("TB")) {
            t10McNemarScoreTB += Math.pow(abs(score10[1] - score10[2]) - 1, 2) / (score10[1] + score10[2]);
        } else if (option.equals("CB")) {
            t10McNemarScoreCB += Math.pow(abs(score10[1] - score10[2]) - 1, 2) / (score10[1] + score10[2]);
        }

    }

    private static double[] calculateVectorDifference(double[] first, double[] second) {

        double[] output = new double[4];
        double a = 0, b = 0, c = 0, d = 0;

        for (int i = 0; i < first.length; i++) {
            if (first[i] == 0.0 && second[i] == 0.0) {
                a++;
            } else if (first[i] == 0.0 && second[i] == 1.0) {
                b++;
            } else if (first[i] == 1.0 && second[i] == 0.0) {
                c++;
            } else if (first[i] == 1.0 && second[i] == 1.0) {
                d++;
            }
        }

        output[0] = a;
        output[1] = b;
        output[2] = c;
        output[3] = d;
        return output;
    }
*/


}