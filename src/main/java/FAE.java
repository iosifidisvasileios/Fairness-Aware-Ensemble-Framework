import org.apache.log4j.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static java.lang.Math.abs;


public class FAE extends AbstractClassifier implements Classifier {

    private final static Logger log = Logger.getLogger(FAE.class.getName());
    private final AdaBoostM1 boost;

    private int k_NumBags = -1;
    private double eqOp = 0.0;
    private double threshold ;
    private int m_SampleNumExecuted ;
    private Classifier m_Boost[];
//    private ArrayList<Double> equalOpp;

    public double getEqOp() {
        return eqOp;
    }

    public void setEqOp(double eqOp) {
        this.eqOp = eqOp;
    }

    private String protectedValueName;
    private int protectedValueIndex;
    private boolean useThreshold;

    private String targetClass;
    private String otherClass;
    private Instances TrainingSet;
    private double bound;
    private double flipProportion;
    private Instances flipedInstances;
    private Instances FR;
    private Instances DG;
    private Instances DR;
    private Instances FG;

    public boolean isParallelization() {
        return parallelization;
    }
    private HashMap<Integer, Instances> clusterAssignments;

    public void setParallelization(boolean parallelization) {
        this.parallelization = parallelization;
    }

    private boolean parallelization = false;

    public double getFlipProportion() {
        return flipProportion;
    }

    public void setFlipProportion(double flipProportion) {
        this.flipProportion = flipProportion;
    }

    public double getRrbVaule() {
        return rrbVaule;
    }

    public void setRrbVaule(double rrbVaule) {
        this.rrbVaule = rrbVaule;
    }

    private double rrbVaule;

    private boolean RRBOption= false;

    public boolean isRRBOption() {
        return RRBOption;
    }

    public void setRRBOption(boolean RRBOption) {
        this.RRBOption = RRBOption;
    }

    public FAE(Classifier baseClassifier,
               int protectedValueIndex,
               String protectedValueName,
               String targetClass,
               String otherClass,
               double bound,
               boolean useThreshold) throws Exception {

        this.useThreshold = useThreshold;
        this.bound = bound;
        this.protectedValueIndex = protectedValueIndex;
        this.protectedValueName = protectedValueName;
        this.targetClass = targetClass;
        this.otherClass = otherClass;

        boost = new AdaBoostM1();
        boost.setNumIterations(25);
        boost.setClassifier(baseClassifier);

    }




    public void buildClassifier(final Instances data) throws Exception {

        data.deleteWithMissingClass();
        TrainingSet = data;

        if(RRBOption){
            addSyntheticBias();
        }
        initializeLists();
        ClusterSummary clusteredFG = initializeClusters(FG);
        ClusterSummary clusteredFR = initializeClusters(FR);
        ClusterSummary clusteredDR = initializeClusters(DR);

        m_Boost = AbstractClassifier.makeCopies(boost, this.k_NumBags);

        if (parallelization) {
            ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() / 2);
            for (int sampleNum = 0; sampleNum < k_NumBags; sampleNum++) {
                final Instances CurrentTraininSet = generateCandidateSetsFromCluster(clusteredFG, clusteredFR, clusteredDR);
//                final Instances CurrentTraininSet = generateCandidateSetsRandomly();

                final int finalSampleNum = sampleNum;
                es.execute(new Runnable() {
                    @Override
                    public void run() {
                        m_SampleNumExecuted = finalSampleNum;
                        try {
                            m_Boost[finalSampleNum].buildClassifier(CurrentTraininSet);
                        } catch (Exception e) {
                        }
                    }
                });
            }
            es.shutdown();
            es.awaitTermination(k_NumBags, TimeUnit.SECONDS);

        }else {

            for (int sampleNum = 0; sampleNum < k_NumBags; sampleNum++) {
//                final Instances CurrentTraininSet = generateCandidateSetsRandomly();
                final Instances CurrentTraininSet = generateCandidateSetsFromCluster(clusteredFG, clusteredFR, clusteredDR);

                m_Boost[sampleNum].buildClassifier(CurrentTraininSet);
            }
        }
//        log.info("calculating EqOp ...");
        double disc = 100*calculateEquallizedOpportunity();
//        log.info("EqOp calculated ! ");

        if (useThreshold) {
            if ((abs(disc) > bound )) {
//                log.info("estimating threshold ...");
                calculateMajorityThreshold(disc);
                eqOp = disc;
//                log.info("threshold estimated !");
            }
        }

        if(RRBOption){
            calculateRBB();
        }
    }

    private Instances generateCandidateSetsFromCluster(ClusterSummary clusteredFG, ClusterSummary clusteredFR, ClusterSummary clusteredDR) {

        Instances output = new Instances(DG);


        for (Instance inst: clusteredFG.getPoints()){
            output.add(inst);
        }

        for (Instance inst: clusteredFR.getPoints()){
            output.add(inst);
        }

        for (Instance inst: clusteredDR.getPoints()){
            output.add(inst);
        }


        return output;
    }

    private ClusterSummary initializeClusters(Instances inputInstances) throws Exception {

        clusterAssignments = new HashMap<Integer,Instances>();
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (inputInstances.classIndex() + 1));
        filter.setInputFormat(inputInstances);
        Instances dataClusterer = Filter.useFilter(inputInstances, filter);



        EM clusterer = new EM();
        clusterer.setMaxIterations(100);
        clusterer.setNumClusters(-1);
        clusterer.buildClusterer(dataClusterer);

        if (clusterer.numberOfClusters() > k_NumBags){
            clusterer = new EM();
            clusterer.setMaxIterations(100);
            clusterer.setNumClusters(k_NumBags);
            clusterer.buildClusterer(dataClusterer);
        }

        for (int i = 0; i < clusterer.numberOfClusters() ; i++) {
            clusterAssignments.put(i, new Instances(inputInstances, 0));
        }

        for (int i=0; i < dataClusterer.size(); i++) {
            clusterAssignments.get(clusterer.clusterInstance(dataClusterer.instance(i))).add(inputInstances.instance(i));
        }

        ClusterSummary clusterSummary = new ClusterSummary();
        clusterSummary.setTotalInstances(inputInstances.size());
        clusterSummary.setBagInstances(DG.size());

        for(int i = 0 ; i < clusterer.numberOfClusters(); i++) {
            clusterSummary.insertCluster(clusterAssignments.get(i), clusterAssignments.get(i).size());
        }
        return clusterSummary;
    }

    private Instances generateCandidateSetsRandomly() {
        Instances output = new Instances(TrainingSet, 0);
        DR.randomize(new Random());
        FG.randomize(new Random());
        FR.randomize(new Random());

        for (int i = 0; i < DG.size(); i++){

            output.add(DG.get(i));
            try {
                output.add(DR.get(i));
            }catch (IndexOutOfBoundsException e){
                output.add(DR.get(new Random().nextInt(DR.size())));
            }
            try {
                output.add(FG.get(i));
            }catch (IndexOutOfBoundsException e){
                output.add(FG.get(new Random().nextInt(FG.size())));
            }
            try {
                output.add(FR.get(i));
            }catch (IndexOutOfBoundsException e){
                output.add(FR.get(new Random().nextInt(FR.size())));
            }
        }
        return  output;
    }

    public void calculateRBB() throws Exception {
        final Evaluation eval = new Evaluation(TrainingSet);

        eval.evaluateModel(this, flipedInstances);
        setRrbVaule(eval.errorRate());
    }

    private void addSyntheticBias() {
        // data are already in random order due to initial shuffling
        int pseudoProtectedCount = 0;
        flipedInstances = new Instances(TrainingSet, 0);
        ArrayList<Integer> tempIndexes = new ArrayList<Integer>();
        int count = 0;
        for (Instance instance: TrainingSet){
            if (!instance.stringValue(protectedValueIndex).equals(protectedValueName) && !instance.stringValue(instance.classIndex()).equals(targetClass)){
                tempIndexes.add(count);
                pseudoProtectedCount += 1;
            }
            count +=1;
        }

        int m = (int)(pseudoProtectedCount*flipProportion);
        for(int i=0; i< m; i++){
            TrainingSet.get(tempIndexes.get(i)).setClassValue(targetClass);
            TrainingSet.get(tempIndexes.get(i)).setClassValue(1.0);
            flipedInstances.add(TrainingSet.get(tempIndexes.get(i)));
        }
    }

    public void done() throws Exception {
        Evaluation fixedEval = new Evaluation(TrainingSet);
        fixedEval.evaluateModel(this, TrainingSet);
        double disc = 100*calculateEquallizedOpportunity();
        log.info("Overall Training Set After change: Accuracy = " + fixedEval.pctCorrect() + ", au-PRC = " + fixedEval.weightedAreaUnderPRC()*100 + ", au-ROC = " + fixedEval.weightedAreaUnderROC()*100+ ", equallized opportunity = " + disc );
    }

    private void calculateMajorityThreshold(double signatureFlag) throws Exception {
        Instances FG = new Instances(TrainingSet, 0);
        Instances DG = new Instances(TrainingSet, 0);

        for (Instance instance : TrainingSet){
            if (!instance.stringValue(protectedValueIndex).equals(protectedValueName) && instance.stringValue(instance.classIndex()).equals(targetClass)) {
                FG.add(instance);
            } else if(instance.stringValue(protectedValueIndex).equals(protectedValueName) && instance.stringValue(instance.classIndex()).equals(targetClass)){
                DG.add(instance);
            }
        }

        int femalePos = DG.size();
        int malePos = FG.size();
        double TPMale = 0.0;
        double TPFemale = 0.0;
        double FNMale = 0.0;
        double FNFemale = 0.0;

        ArrayList<Double> probabilitiesMale = new ArrayList<Double>();
        ArrayList<Double> probabilitiesFemale = new ArrayList<Double>();

        for (Instance instance : FG){
            if (this.classifyInstance(instance) == instance.classValue()){
                TPMale +=1;
            }else{
                FNMale +=1;
                probabilitiesMale.add(this.distributionForInstance(instance)[1]);
            }
        }

        for (Instance instance : DG){
            if (this.classifyInstance(instance) == instance.classValue()){
                TPFemale +=1;
            }else{
                FNFemale +=1;
                probabilitiesFemale.add(this.distributionForInstance(instance)[1]);

            }
        }

        if (signatureFlag < 0) {
            int y = (int) ((TPFemale / (femalePos)) * malePos - TPMale) ;
            if (y == probabilitiesMale.size())
                y -= 1;

            Collections.sort(probabilitiesMale);
            Collections.reverse(probabilitiesMale);
            threshold = probabilitiesMale.get(y) + .00000000000001;

        }else if(signatureFlag > 0){
            int y = (int) ((TPMale / (malePos)) * femalePos - TPFemale);
            if (y == probabilitiesFemale.size())
                y -= 1;

            Collections.sort(probabilitiesFemale);
            Collections.reverse(probabilitiesFemale);
            threshold = probabilitiesFemale.get(y) + .00000000000001;
        }
    }

    private void initializeLists() {

        TrainingSet.randomize(new Random());

        DG = new Instances(TrainingSet, 0);
        FG = new Instances(TrainingSet, 0);
        DR = new Instances(TrainingSet, 0);
        FR = new Instances(TrainingSet, 0);

        Instances output = new Instances(TrainingSet, 0);

        for (Instance instance : TrainingSet){
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

        int maximum = -1;
        if (DR.size() > maximum)
            maximum = DR.size();
        if (FR.size() > maximum)
            maximum = FR.size();
        if (FG.size() > maximum)
            maximum = FG.size();

        int bags= (int)(1/((double) DG.size()/(double) maximum)) + 1;
//        log.info("maximum instances = " + DG.size());
//        log.info("bags = " + bags);
        k_NumBags = bags;

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {

        double[] dist = distributionForInstance(instance);
        if (dist == null) {
            throw new Exception("Null distribution predicted");
        }
        switch (instance.classAttribute().type()) {

            case Attribute.NOMINAL:
                double max = 0;
                int maxIndex = 0;

                for (int i = 0; i < dist.length; i++) {
                    if (dist[i] > max) {
                        maxIndex = i;
                        max = dist[i];
                    }
                }
                if (max > 0) {
                    return maxIndex;
                } else {
                    return Utils.missingValue();
                }
            case Attribute.NUMERIC:
            case Attribute.DATE:
                return dist[0];
            default:
                return Utils.missingValue();
        }
    }

    @Override
    public double [] distributionForInstance(Instance ins) throws Exception {

        double sum[] = new double[ins.numClasses()];
        for (int sampleNum = 0; sampleNum < k_NumBags; sampleNum++) {

            try {
                double results[] = m_Boost[sampleNum].distributionForInstance(ins);
                sum[0] = sum[0] + results[0];
                sum[1] = sum[1] + results[1];
            }catch (IllegalArgumentException e){
                sum[0] += 0.1;
                sum[1] += 0.1;
            }
        }

        Utils.normalize(sum);
        if (eqOp < 0) {
            if (sum[1] >= threshold && !ins.stringValue(protectedValueIndex).equals(protectedValueName)){
                sum[0] = 0;
                sum[1] = 1;
            }
        }else if(eqOp > 0){
            if (sum[1] >= threshold && ins.stringValue(protectedValueIndex).equals(protectedValueName)){
                sum[0] = 0;
                sum[1] = 1;
            }
        }
        return sum;
    }

    private double calculateEquallizedOpportunity() throws Exception {
//        Instances TestingPredictions = new Instances(TrainingSet);

        double tp_male = 0;
        double tn_male = 0;
        double tp_female = 0;
        double tn_female = 0;
        double fp_male = 0;
        double fn_male = 0;
        double fp_female = 0;
        double fn_female = 0;

        for(Instance ins: TrainingSet){
            double label = this.classifyInstance(ins);
            if (ins.stringValue(protectedValueIndex).equals(protectedValueName)) {
                if (label == ins.classValue()) {
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_female++;
                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        tn_female++;
                    }
                }else{
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        fn_female++;
                    }else if (ins.stringValue(ins.classIndex()).equals(otherClass)){
                        fp_female++;
                    }
                }
            }else{
                if (label == ins.classValue()) {
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_male++;
                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        tn_male++;
                    }
                }else{
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        fn_male++;
                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        fp_male++;
                    }
                }
            }
        }
        return (tp_male)/(tp_male + fn_male) - (tp_female)/(tp_female + fn_female);
    }

}
