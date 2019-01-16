import org.apache.log4j.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.DecisionStump;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

/**
 * Created by iosifidis on 13.08.18.
 */
public class RrbEvaluation {

    private final static Logger log = Logger.getLogger(RrbEvaluation.class.getName());
    private static String protectedValueName;
    private static int protectedValueIndex;
    private static String targetClass;
    private static String otherClass;
    private static Classifier model;

    private FAE boost;


    public RrbEvaluation(Classifier f,
                         int protectedValueIndex,
                         String protectedValueName,
                         String targetClass,
                         String otherClass,
                         Instances data,
                         double bound,
                         double flipPercentage,
                         boolean thresholdUse) throws Exception {

        boost = new FAE(f, protectedValueIndex, protectedValueName, targetClass, otherClass, bound, thresholdUse);
        boost.setRRBOption(true);
        boost.setFlipProportion(flipPercentage);
        boost.buildClassifier(data);

    }

    public double getRrb() {
        return boost.getRrbVaule() * 100;
    }

    public FAE getBoost() {
        return boost;
    }

    public static ArrayList<Double> FAE_RBB = new ArrayList<Double>();
    public static ArrayList<Double> OB_RBB = new ArrayList<Double>();
    public static ArrayList<Double> SMT_RBB = new ArrayList<Double>();
    public static ArrayList<Double> SDB_RBB = new ArrayList<Double>();

    public static void main(String [] argv) throws Exception {
//        String modelString = argv[0];
//        final String parameters = argv[1];
//        final double bound = Double.valueOf(argv[2]);
//        final BufferedReader reader = new BufferedReader(new FileReader(argv[3]));

        String modelString = "NB";
        String parameters = "propublica";
        double bound = 0.0;
        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/compass_zafar.arff"));
//        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff"));
//        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/dutch.arff"));
//        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/kdd.arff"));
        log.info("dataset = " + parameters);
        if(modelString.equals("NB")){
            model = new NaiveBayes();
        } else if(modelString.equals("DS")){
            model = new DecisionStump();
        }else if(modelString.equals("LR")){
            Logistic xxx = new Logistic();
            if(parameters.equals("kdd")) {
                // big dataset use less iterations and gradient
                xxx.setMaxIts(5);
//                xxx.setUseConjugateGradientDescent(true);
            }
            model = xxx;
        } else{
            System.exit(1);
        }

        if (parameters.equals("adult-gender")){
            protectedValueName = " Female";
            protectedValueIndex = 8;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if(parameters.equals("adult-race")){
            protectedValueName = " Minorities";
            protectedValueIndex = 7;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if(parameters.equals("dutch")){
            protectedValueName = "2"; // women
            protectedValueIndex = 0;
            targetClass = "2_1"; // high level
            otherClass = "5_4_9";
        } else if (parameters.equals("kdd")){
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
            protectedValueName = "0";
            protectedValueIndex = 4;
            targetClass = "1";
            otherClass = "-1";
//            protectedValueName = "Female";
//            protectedValueIndex =2;
//            targetClass = "1";
//            otherClass = "0";
        } else{
            System.exit(1);

        }


        final Instances data = new Instances(reader);
        reader.close();

        for (double flipPercentage = 0.1; flipPercentage <= 0.61; flipPercentage += 0.1) {
            for (int k = 0; k < 5; k++) {
//                log.info("k = " + k);
                final Random rand = new Random((int) System.currentTimeMillis());   // create seeded number generator
                final Instances randData = new Instances(data);   // create copy of original data
                randData.randomize(rand);         // randomize data with number generator
                randData.setClassIndex(data.numAttributes() - 1);

                int trainSize = (int) Math.round(randData.numInstances() * 0.677);
//                int testSize = randData.numInstances() - trainSize;
                Instances train = new Instances(randData, 0, trainSize);
//                Instances test = new Instances(randData, trainSize, testSize);

                RrbEvaluation FBF = new RrbEvaluation(AbstractClassifier.makeCopy(model),
                        protectedValueIndex,
                        protectedValueName,
                        targetClass,
                        otherClass,
                        new Instances(train),
                        bound,
                        flipPercentage,
                        true);
                FAE_RBB.add(FBF.getRrb());
//                log.info("finished FAE");

                RrbEvaluation EasyEnsemble = new RrbEvaluation(AbstractClassifier.makeCopy(model),
                        protectedValueIndex,
                        protectedValueName,
                        targetClass,
                        otherClass,
                        new Instances(train),
                        bound,
                        flipPercentage,
                        false);

                OB_RBB.add(EasyEnsemble.getRrb());
//                log.info("finished OB");

                SMT thresholdBoosting = new SMT(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound);
                thresholdBoosting.setNumIterations(25);
                thresholdBoosting.setRRBOption(true);
                thresholdBoosting.setFlipProportion(flipPercentage);
                thresholdBoosting.buildClassifier(new Instances(train));
                SMT_RBB.add(thresholdBoosting.getRrb());

                SDB confidenceBoosting = new SDB (AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound);
                confidenceBoosting.setNumIterations(25);
                confidenceBoosting.setRRBOption(true);
                confidenceBoosting.setFlipProportion(flipPercentage);
                confidenceBoosting.buildClassifier(new Instances(train));
                SDB_RBB.add(confidenceBoosting.getRrb());

            }


            calculateSD(FAE_RBB, "FAE RBB for n = " + flipPercentage + ",");
            calculateSD(OB_RBB, "OB RBB for n = " + flipPercentage + ",");

            calculateSD(SMT_RBB, "SMT RBB for n = " + flipPercentage + ",");
            calculateSD(SDB_RBB, "SDB RBB for n = " + flipPercentage + ",");
            log.info("*******************************");
            FAE_RBB.clear();
            OB_RBB.clear();
            SMT_RBB.clear();
            SDB_RBB.clear();
        }
    }

    private static void calculateSD(ArrayList<Double> numArray, String measurement) {
        double sum = 0.0, standardDeviation = 0.0;
        for(double num : numArray) {
            sum += num;
        }
        double mean = sum/numArray.size();
        for(double num: numArray) {
            standardDeviation = standardDeviation + Math.pow(num - mean, 2)/numArray.size();
        }
        standardDeviation = Math.sqrt(standardDeviation);
        log.info(measurement + " mean = " + mean  + ", st.Dev = " + standardDeviation);
    }
}