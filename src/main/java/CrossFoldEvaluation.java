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

import static java.lang.Math.abs;

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


    public static ArrayList<Double> FBFaccuracy = new ArrayList<Double>();
    public static ArrayList<Double> FBFauPRC = new ArrayList<Double>();
    public static ArrayList<Double> FBFauROC = new ArrayList<Double>();
    public static ArrayList<Double> FBFf1weighted = new ArrayList<Double>();
    public static ArrayList<Double> FBFkappa = new ArrayList<Double>();
    public static ArrayList<Double> FBFdiscList = new ArrayList<Double>();

    public static ArrayList<Double> Easyaccuracy = new ArrayList<Double>();
    public static ArrayList<Double> EasyauPRC = new ArrayList<Double>();
    public static ArrayList<Double> EasyauROC = new ArrayList<Double>();
    public static ArrayList<Double> Easyf1weighted = new ArrayList<Double>();
    public static ArrayList<Double> EasydiscList = new ArrayList<Double>();
    public static ArrayList<Double> Easykappa = new ArrayList<Double>();


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



    public static double t10McNemarScoreTB = 0;
    public static double testForSE = 0;

    public static double t10McNemarScoreCB = 0;

    public static ArrayList<Double> discListthresholdBoosting = new ArrayList<Double>();
    public static ArrayList<Double> discListconfidenceBoosting = new ArrayList<Double>();

    public static void main(String [] argv) throws Exception {

/*        String modelString = argv[0];
        final String parameters = argv[1];
        final double bound = Double.valueOf(argv[2]);
        final BufferedReader reader = new BufferedReader(new FileReader(argv[3]));
*/
        String modelString = "LR";
        String parameters = "kdd";
        double bound = 0.0;
//        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/violent_compassProcessed.arff"));
//        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/compass_zafar.arff"));
//        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/adult.arff"));
//        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/dutch.arff"));
        final BufferedReader reader = new BufferedReader(new FileReader("/home/iosifidis/BiasForStreams/out/artifacts/BiasForStreams_jar/kdd.arff"));

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
        }else{
            System.exit(1);
        }
        int folds = 5;
        final Instances data = new Instances(reader);
        reader.close();
        int iterations = 10;
//        provideStatistics(data);
//        System.exit(1);
        log.info("dataset = " + parameters );
        int k = 0;
        while (k < iterations){
//        for (int k=0; k< iterations ; k++) {
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

//            for (int n = 0; n < folds; n++) {
//                log.info("running fold = " + n);
//                Instances train = randData.trainCV(folds, n);
//                Instances test = randData.testCV(folds, n);

                final CrossFoldEvaluation FAEF = new CrossFoldEvaluation(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, train, bound, true);
//                final CrossFoldEvaluation easyEns = new CrossFoldEvaluation(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, train, bound, false);
                final SMT thresholdBoosting = new SMT(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound);
                final SDB confidenceBoosting = new SDB(AbstractClassifier.makeCopy(model), protectedValueIndex, protectedValueName, targetClass, otherClass, bound);
                thresholdBoosting.setNumIterations(25);
                thresholdBoosting.buildClassifier(train);
                log.info("SMT Training");
                confidenceBoosting.setNumIterations(25);
                confidenceBoosting.buildClassifier(train);
                log.info("SDB Training");

                double[] measures = EvaluateClassifier(FAEF.getBoost(), train, test);
                double disc = equalOpportunityMeasurement(FAEF.getBoost(), test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                FBFaccuracy.add(measures[0]);
                FBFauPRC.add(measures[1]);
                FBFauROC.add(measures[2]);
                FBFf1weighted.add(measures[3]);
                FBFkappa.add(measures[4]);
                FBFdiscList.add(disc);
                double[] FBFpredictions = getBinaryPredictions(FAEF.getBoost(), test);
//            log.info("finished FAE");

                // if EqOp is 0 then no special treatment is employed, this equals to OB model
                FAEF.getBoost().setEqOp(0.0);
                measures = EvaluateClassifier(FAEF.getBoost(), train, test);
                disc = equalOpportunityMeasurement(FAEF.getBoost(), test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                Easyaccuracy.add(measures[0]);
                EasyauPRC.add(measures[1]);
                EasyauROC.add(measures[2]);
                Easyf1weighted.add(measures[3]);
                Easykappa.add(measures[4]);
                EasydiscList.add(disc);
                double[] Easypredictions = getBinaryPredictions(FAEF.getBoost(), test);
//            log.info("finished Easy");

                measures = EvaluateClassifier(thresholdBoosting, train, test);
                disc = equalOpportunityMeasurement(thresholdBoosting, test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                accuracythresholdBoosting.add(measures[0]);
                auPRCthresholdBoosting.add(measures[1]);
                auROCthresholdBoosting.add(measures[2]);
                f1weightedthresholdBoosting.add(measures[3]);
                kappaweightedthresholdBoosting.add(measures[4]);
                discListthresholdBoosting.add(disc);
                double[] TBpredictions = getBinaryPredictions(thresholdBoosting, test);

//            calculateMcNemarSignificantScore(TBpredictions, t5predictions, t10predictions, t50predictions, t100predictions, t200predictions,  "TB");
                measures = EvaluateClassifier(confidenceBoosting, train, test);
                disc = equalOpportunityMeasurement(confidenceBoosting, test, protectedValueIndex, protectedValueName, targetClass, otherClass);
                accuracyconfidenceBoosting.add(measures[0]);
                auPRCconfidenceBoosting.add(measures[1]);
                auROCconfidenceBoosting.add(measures[2]);
                f1weightedconfidenceBoosting.add(measures[3]);
                kappaweightedconfidenceBoosting.add(measures[4]);
                discListconfidenceBoosting.add(disc);
                double[] CBpredictions = getBinaryPredictions(confidenceBoosting, test);

                //            calculateMcNemarSignificantScore(CBpredictions , t5predictions, t10predictions, t50predictions, t100predictions,t200predictions,  "CB");
//            outputPredictionsToFile(TBpredictions, CBpredictions ,Easypredictions,  FBFpredictions, parameters, modelString);
//            outputPredictionsToFile(FBFpredictions, Easypredictions, parameters, modelString);
//            }
            }catch (Exception e){
                log.info("crushed, rerun iteration");
                k--;
                continue;
            }

        }

        testForSE = testForSE / iterations;
        log.info("dataset = " + parameters);
//        calculateSD(FBFaccuracy, "FAE Accuracy");
//        calculateSD(FBFauPRC, "FAE Au-PRC");
        calculateSD(FBFauROC, "FAE Au-ROC");
//        calculateSD(FBFf1weighted, "FAE F1");
//        calculateSD(FBFkappa, "FAE kappa");
        calculateSD(FBFdiscList, "FAE disc");

//        calculateSD(Easyaccuracy, "Easy Accuracy");
//        calculateSD(EasyauPRC, "Easy Au-PRC");
        calculateSD(EasyauROC, "OB Au-ROC");
//        calculateSD(Easyf1weighted, "Easy F1");
//        calculateSD(Easykappa, "Easy kappa");
        calculateSD(EasydiscList, "OB disc");

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

    }

    private static void outputPredictionsToFile(double [] SMT, double [] SDB, double[] OB, double[] BFF, String parameters, String modelString) throws IOException {



        FileWriter fstream = new FileWriter("significance/" + parameters +"_" + modelString + ".csv", true); //true tells to append data.
        BufferedWriter out = new BufferedWriter(fstream);
        out.write("index\tSDB\tSMT\tOB\tBFF\n");

        for (int i=0;i < BFF.length; i++){
            out.write(i + "\t" + SDB[i] + "\t" + SMT[i] +"\t" + OB[i] +"\t" + BFF[i] + "\n");
        }

        out.close();

    }

    private static void provideStatistics(Instances data) {
        data.setClassIndex(data.numAttributes() - 1);

        ArrayList<Instance> DG = new ArrayList<Instance>();
        ArrayList<Instance> DR = new ArrayList<Instance>();
        ArrayList<Instance> FG = new ArrayList<Instance>();
        ArrayList<Instance> FR = new ArrayList<Instance>();
        for (Instance instance : data){
            if (instance.stringValue(protectedValueIndex).equals(protectedValueName)) {
                if (instance.stringValue(instance.classIndex()).equals(targetClass)) {

                    DG.add(instance);
                }else if(instance.stringValue(instance.classIndex()).equals(otherClass)){
                    DR.add(instance);
                }
            }else {
                if (instance.stringValue(instance.classIndex()).equals(targetClass)) {
                    FG.add(instance);
                }else if(instance.stringValue(instance.classIndex()).equals(otherClass)){
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

    private static void calculateMcNemarSignificantScore(double[] ComparingPredictions,
                                                         double[] t5predictions,
                                                         double[] t10predictions,
//                                                         double[] t20predictions,
                                                         double[] t50predictions,
                                                         double[] t100predictions,
                                                         double[] t200predictions,
//                                                         double[] t500predictions,
                                                         String option) {

        double[] score5 = calculateVectorDifference(ComparingPredictions, t5predictions);
        double[] score10 = calculateVectorDifference(ComparingPredictions, t10predictions);
//        double[] score20 = calculateVectorDifference(ComparingPredictions, t20predictions);
        double[] score50 = calculateVectorDifference(ComparingPredictions, t50predictions);
        double[] score100 = calculateVectorDifference(ComparingPredictions, t100predictions);
        double[] score200 = calculateVectorDifference(ComparingPredictions, t200predictions);
//        double[] score500 = calculateVectorDifference(ComparingPredictions, t500predictions);

        if (option.equals("TB")) {
            t10McNemarScoreTB += Math.pow(abs(score10[1] - score10[2]) - 1, 2) / (score10[1] + score10[2]);
        }else if(option.equals("CB")){
            t10McNemarScoreCB += Math.pow(abs(score10[1] - score10[2]) - 1, 2) / (score10[1] + score10[2]);
        }
    }

    private static double[] calculateVectorDifference(double[] first, double[] second) {

        double [] output = new double[4];
        double a = 0, b = 0, c = 0, d = 0;

        for (int i=0; i< first.length; i++){
            if (first[i] == 0.0 && second[i] == 0.0){
                a ++;
            }else if(first[i] == 0.0 && second[i] == 1.0){
                b ++;
            }else if(first[i] == 1.0 && second[i] == 0.0){
                c ++;
            }else if(first[i] == 1.0 && second[i] == 1.0){
                d ++;
            }
        }
        output[0] = a;
        output[1] = b;
        output[2] = c;
        output[3] = d;
        return output;
    }

    private static double[] getBinaryPredictions(Classifier clf, Instances test) throws Exception {
        double preds[] = new double[test.size()];
        int index = 0;
        for (Instance instance: test){
            preds[index] = clf.classifyInstance(instance);
            index ++;
        }

        return preds;
    }


    public static void calculateSD(ArrayList<Double> numArray, String measurement) {
        double sum = 0.0, standardDeviation = 0.0;

        for(double num : numArray) {
            sum += num;
        }

        double mean = sum/numArray.size();

        for(double num: numArray) {
            standardDeviation = standardDeviation + Math.pow(num - mean, 2)/numArray.size();
        }

        standardDeviation = Math.sqrt(standardDeviation);
//        log.info(measurement + " mean = " + mean  + ", st.Dev = " + standardDeviation);


        double standardError = standardDeviation/Math.sqrt(testForSE);
        log.info(measurement + " mean = " + mean  + ", St.Error = " + standardError +  ", st.Dev = " + standardDeviation);


    }


    public static double [] EvaluateClassifier(Classifier classifier, Instances training, Instances testing) throws Exception {

        final Evaluation eval = new Evaluation(training);
        eval.evaluateModel(classifier, testing);
        return new double []{eval.pctCorrect(), eval.weightedAreaUnderPRC()*100, eval.weightedAreaUnderROC()*100, eval.weightedFMeasure()*100, eval.kappa()*100};
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


        for(Instance ins: TestingPredictions){
            double label = classifier.classifyInstance(ins);
            // WOMEN
            if (ins.stringValue(protectedValueIndex).equals(protectedValueName)) {
                // correctly classified
                if (label == ins.classValue()) {
                    correctlyClassified++;

                    // on target class (true negatives)
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_female++;

                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        tn_female++;
                    }
                }else{
                    // error has been made on TN so it's FP
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        fn_female++;

                    }else if (ins.stringValue(ins.classIndex()).equals(otherClass)){
                        // error has been made on TP so it's FN
                        fp_female++;

                    }
                }
            }else{
                // correctly classified
                if (label == ins.classValue()) {
                    correctlyClassified++;
                    // on target class (true negatives)
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_male++;

                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        tn_male++;

                    }
                }else{
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        // error has been made on TP so it's FN

                        fn_male++;

                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        // error has been made on TN so it's FP

                        fp_male++;

                    }
                }
            }
        }
        return 100*((tp_male)/(tp_male + fn_male) - (tp_female)/(tp_female + fn_female));
    }
}

// dataset = ProPublica
//    FBF Au-ROC mean = 66.71486154363569, St.Error = Infinity, st.Dev = 1.584158487815468
//        FBF disc mean = -3.066350910954441, St.Error = Infinity, st.Dev = 4.883651476257302
//        Easy Au-ROC mean = 69.35363730315699, St.Error = Infinity, st.Dev = 1.4211914536232697
//        Easy disc mean = -5.787586003272988, St.Error = Infinity, st.Dev = 5.490065591618045
//        TB Au-ROC mean = 69.95630269246978, St.Error = Infinity, st.Dev = 1.5113907564817308
//        TB disc mean = -13.395423107606952, St.Error = Infinity, st.Dev = 6.534889766062558
//        CB Au-ROC mean = 66.23932691794516, St.Error = Infinity, st.Dev = 1.3014997608726153
//        CB disc mean = -29.952565769343572, St.Error = Infinity, st.Dev = 5.0188673556250825
// dataset = dutch
//        FBF Au-ROC mean = 87.81917759248188, St.Error = 0.001482554950942794, st.Dev = 0.20711244908729737
//        FBF disc mean = -9.332839885939348, St.Error = 0.006020767286697515, st.Dev = 0.8410992505469166
//        Easy Au-ROC mean = 87.86690617647564, St.Error = 0.001702233032424025, st.Dev = 0.23780140630769805
//        Easy disc mean = -9.38187037479108, St.Error = 0.006023576482723861, st.Dev = 0.8414916943269011
//        TB Au-ROC mean = 86.63492829868133, St.Error = 0.0026214088048598263, st.Dev = 0.3662099656328174
//        TB disc mean = 3.4896852175953215, St.Error = 0.006356062877114096, st.Dev = 0.8879399365229664
//        CB Au-ROC mean = 57.73258456173075, St.Error = 0.0016389059397737063, st.Dev = 0.2289546318633208
//        CB disc mean = -16.328611688463525, St.Error = 0.004752960078103175, st.Dev = 0.6639869918913277
// dataset = adult-race
//        FBF Au-ROC mean = 83.4223891802996, St.Error = 0.0019566172174859176, st.Dev = 0.2694852239196746
//        FBF disc mean = 2.108550094452242, St.Error = 0.014407573185787666, st.Dev = 1.9843575183805893
//        Easy Au-ROC mean = 88.75372772494885, St.Error = 0.0018201579057552553, st.Dev = 0.2506906595822958
//        Easy disc mean = -3.4435777148542344, St.Error = 0.0097544358183429, st.Dev = 1.3434801131382512
//        TB Au-ROC mean = 84.45909580071732, St.Error = 0.0017274414382514904, st.Dev = 0.23792080465972806
//        TB disc mean = -8.459516947775978, St.Error = 0.021947589205327852, st.Dev = 3.022845213993615
//        CB Au-ROC mean = 77.08091118351066, St.Error = 0.0024389181813995933, st.Dev = 0.3359126181465105
//        CB disc mean = -8.459516947775978, St.Error = 0.021947589205327852, st.Dev = 3.022845213993615

// dataset = adult-gender
//        FAE Au-ROC mean = 83.42750118084658, St.Error = 0.0023373691254682393, st.Dev = 0.28234799858238063
//        FAE disc mean = 1.1724480254637549, St.Error = 0.011565807870557292, st.Dev = 1.3971189525257426
//        Easy Au-ROC mean = 86.33466238615682, St.Error = 0.0017637410597799307, st.Dev = 0.21305524780843915
//        Easy disc mean = -4.908496077605571, St.Error = 0.012884018100439532, st.Dev = 1.5563552563096001
//        TB Au-ROC mean = 84.32378448628137, St.Error = 0.00264568543984548, st.Dev = 0.3195917926182375
//        TB disc mean = -2.676752100126152, St.Error = 0.015638008903074364, st.Dev = 1.8890300498480221
//        CB Au-ROC mean = 76.8660175154603, St.Error = 0.002738319960224725, st.Dev = 0.33078179728790175
//        CB disc mean = -2.6099582069963807, St.Error = 0.015036184088150634, st.Dev = 1.8163312064606363


// dataset = kdd
//        FAE Au-ROC mean = 91.5513960713329, St.Error = 3.796455057209876E-4, st.Dev = 0.11803800915958099
//        FAE disc mean = 1.6598754606473385, St.Error = 0.003899463520737391, st.Dev = 1.2124071109550516
//        Easy Au-ROC mean = 92.69617717147676, St.Error = 4.579501578902128E-4, st.Dev = 0.1423842087344594
//        Easy disc mean = -0.08486870307277017, St.Error = 0.003138720933499153, st.Dev = 0.9758797739331707
//        TB Au-ROC mean = 80.0370087666977, St.Error = 0.005080426031387583, st.Dev = 1.5795877085088577
//        TB disc mean = -49.22690463889242, St.Error = 0.004463888897623834, st.Dev = 1.387896210135342
//        CB Au-ROC mean = 72.22728883805206, St.Error = 0.0011963338556824179, st.Dev = 0.3719598006218451
//        CB disc mean = -16.396957615512864, St.Error = 0.045660438075989504, st.Dev = 14.196578457075562