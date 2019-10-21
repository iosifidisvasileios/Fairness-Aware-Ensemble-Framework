import org.apache.log4j.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
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
    private AdaBoostM1 adaBoost;

    private int k_NumBags = -1;


    public static final int EM_CLUSTERING = 0;
    public static final int KNN_WITH_ELBOW = 1;
    private int bagsForOB;
    private boolean optimize = true;

    public int getDECIDE_CLUSTER_METHOD() {
        return DECIDE_CLUSTER_METHOD;
    }

    public void setDECIDE_CLUSTER_METHOD(int DECIDE_CLUSTER_METHOD) {
        this.DECIDE_CLUSTER_METHOD = DECIDE_CLUSTER_METHOD;
    }

    /** Validation method to use. */
    protected int DECIDE_CLUSTER_METHOD = KNN_WITH_ELBOW;

    public int getMaxClusterIteration() {
        return maxClusterIteration;
    }

    public void setMaxClusterIteration(int maxClusterIteration) {
        this.maxClusterIteration = maxClusterIteration;
    }

    private int maxClusterIteration = 100;
    private double eqOp = 0.0;
    private double threshold = 0.5;
    private ArrayList<Classifier> m_Boost = new ArrayList<>();

    public double getEqOp() {
        return eqOp;
    }

    public void setEqOp(double eqOp) {
        this.eqOp = eqOp;
        this.k_NumBags = bagsForOB;
    }

    private String protectedValueName;
    private int protectedValueIndex;
    private boolean useThreshold;

    private String targetClass;
    private String otherClass;
    private Instances TrainingSet;
    private double bound;
    private  Instances FR;
    private  Instances DG;
    private  Instances DR;
    private  Instances FG;

    private ClusterSummary clusteredFG;
    private ClusterSummary clusteredFR;
    private ClusterSummary clusteredDR;

    public boolean isParallelization() {
        return parallelization;
    }
    private HashMap<Integer, Instances> clusterAssignments;

    public void setParallelization(boolean parallelization) {
        this.parallelization = parallelization;
    }

    private boolean parallelization = false;
    private boolean userDefinedBags= false;

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

        adaBoost = new AdaBoostM1();
        adaBoost.setNumIterations(25);
        adaBoost.setClassifier(baseClassifier);
    }



    public int getK_NumBags() {
        return k_NumBags;
    }



    public void buildClassifier(final Instances data) throws Exception {

        data.deleteWithMissingClass();
        TrainingSet = data;


        log.info("initializing clusters");

        initializeLists();
        if (DECIDE_CLUSTER_METHOD == 0) {
            clusteredFG = initializeEM_Clusters(FG);
            log.info("non-prot pos, clustered");
            clusteredFR = initializeEM_Clusters(FR);
            log.info("non-prot neg, clustered");
            clusteredDR = initializeEM_Clusters(DR);
            log.info("prot neg, clustered");

        } else if (DECIDE_CLUSTER_METHOD == 1) {
            clusteredFG = initializeKNN_Clusters(FG);
            log.info("non-prot pos, clustered");

            clusteredFR = initializeKNN_Clusters(FR);
            log.info("non-prot neg, clustered");
            clusteredDR = initializeKNN_Clusters(DR);
            log.info("prot neg, clustered");


        }


        log.info("clusters initialized");
        for (int j = 0; j < k_NumBags; j++) {
            m_Boost.add(AbstractClassifier.makeCopy(adaBoost));
        }

        if (parallelization) {
            ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            for (int sampleNum = 0; sampleNum < k_NumBags; sampleNum++) {
                final Instances CurrentTraininSet = generateCandidateSetsFromCluster(clusteredFG, clusteredFR, clusteredDR);
                final int finalSampleNum = sampleNum;
                es.execute(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            m_Boost.get(finalSampleNum).buildClassifier(CurrentTraininSet);
                        } catch (Exception e) {
                        }
                    }
                });
            }
            es.shutdown();
            es.awaitTermination(k_NumBags, TimeUnit.SECONDS);

        } else {

            for (int sampleNum = 0; sampleNum < k_NumBags; sampleNum++) {
                final Instances CurrentTraininSet = generateCandidateSetsFromCluster(this.clusteredFG, this.clusteredFR, this.clusteredDR);
                m_Boost.get(sampleNum).buildClassifier(CurrentTraininSet);
            }
        }
        log.info("training of bags finished. now proceed to find best sequence");

        bagsForOB = k_NumBags;


        if (this.optimize){

            ArrayList<Double> scores = new ArrayList<>();
            ArrayList<Double> eqopList = new ArrayList<>();
            ArrayList<Double> thresList = new ArrayList<>();
            final int numberOfBagsToCheck = k_NumBags;
            int start = 1;
            for (int counter = start; counter <= numberOfBagsToCheck; counter++) {
                log.info("iteration " + counter + ", out of " + numberOfBagsToCheck);

                if (counter <= numberOfBagsToCheck / 2) {
                    thresList.add(.5);
                    eqopList.add(0.);
                    scores.add(100.);
                    continue;
                }
                k_NumBags = counter;
                double disc = statistics()[0];
                if ((abs(disc) > bound)) {
                    calculateMajorityThreshold(disc);
                    eqOp = disc;
                } else {
                    eqOp = 0.;
                    threshold = 0.5;
                }
                thresList.add(threshold);
                eqopList.add(disc);

                double[] roundScores = statistics();
                // give extra weight to fairness
                scores.add(2 * abs(roundScores[0]) + abs(roundScores[1]));
                eqOp = 0;
            }


            log.info(scores);
            // index start from 0 until the index provided (but without considering it so add +1)
            k_NumBags = scores.indexOf(Collections.min(scores));
            // this does not use +1 so remove 1
            eqOp = eqopList.get(k_NumBags);
            threshold = thresList.get(k_NumBags);
            log.info("best bags = " + (k_NumBags + 1));
        }else{
            double disc = statistics()[0];
            calculateMajorityThreshold(disc);
            eqOp = disc;
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

    private ClusterSummary initializeEM_Clusters(Instances inputInstances) throws Exception {

        clusterAssignments = new HashMap<>();
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (inputInstances.classIndex() + 1));
        filter.setInputFormat(inputInstances);
        Instances dataClusterer = Filter.useFilter(inputInstances, filter);


        EM clusterer = new EM();
        clusterer.setMaxIterations(maxClusterIteration);
        clusterer.setNumClusters(-1);
        clusterer.buildClusterer(dataClusterer);

        if (clusterer.numberOfClusters() > k_NumBags){
            clusterer = new EM();
            clusterer.setMaxIterations(maxClusterIteration);
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


    private ClusterSummary initializeKNN_Clusters(Instances inputInstances) throws Exception {

        clusterAssignments = new HashMap<>();
        weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
        filter.setAttributeIndices("" + (inputInstances.classIndex() + 1));
        filter.setInputFormat(inputInstances);
        Instances dataClusterer = Filter.useFilter(inputInstances, filter);


        KValid clusterer = new KValid();
        clusterer.setMaxIterations(maxClusterIteration);
        clusterer.buildClusterer(dataClusterer);


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



    private void calculateMajorityThreshold(double signatureFlag) throws Exception {
//        Instances FGtemp = new Instances(TrainingSet, 0);
//        Instances DGtemp = new Instances(TrainingSet, 0);

//        for (Instance instance : TrainingSet){
//            if (!instance.stringValue(protectedValueIndex).equals(protectedValueName) && instance.stringValue(instance.classIndex()).equals(targetClass)) {
//                FGtemp.add(instance);
//            } else if(instance.stringValue(protectedValueIndex).equals(protectedValueName) && instance.stringValue(instance.classIndex()).equals(targetClass)){
//                DGtemp.add(instance);
//            }
//        }

        int protectedPos = DG.size();
        int non_protectedPos = FG.size();
        double TPnon_protected = 0.0;
        double TPprotected = 0.0;
        double FNnon_protected = 0.0;
        double FNprotected = 0.0;

        ArrayList<Double> probabilitiesnon_protected = new ArrayList<Double>();
        ArrayList<Double> probabilitiesprotected = new ArrayList<Double>();

        for (Instance instance : FG){
            if (this.classifyInstance(instance) == instance.classValue()){
                TPnon_protected +=1;
            }else{
                FNnon_protected +=1;
                probabilitiesnon_protected.add(this.distributionForInstance(instance)[1]);
            }
        }

        for (Instance instance : DG){
            if (this.classifyInstance(instance) == instance.classValue()){
                TPprotected +=1;
            }else{
                FNprotected +=1;
                probabilitiesprotected.add(this.distributionForInstance(instance)[1]);

            }
        }

        if (signatureFlag < 0) {
            int y = (int) ((TPprotected / (protectedPos)) * non_protectedPos - TPnon_protected) ;
            if (y == probabilitiesnon_protected.size())
                y -= 1;

            Collections.sort(probabilitiesnon_protected);
            Collections.reverse(probabilitiesnon_protected);
            threshold = probabilitiesnon_protected.get(y) + .00000000000001;

        }else if(signatureFlag > 0){
            int y = (int) ((TPnon_protected / (non_protectedPos)) * protectedPos - TPprotected);
            if (y == probabilitiesprotected.size())
                y -= 1;

            Collections.sort(probabilitiesprotected);
            Collections.reverse(probabilitiesprotected);
            threshold = probabilitiesprotected.get(y) + .00000000000001;
        }
    }

    private void initializeLists() {

        TrainingSet.randomize(new Random());

        DG = new Instances(TrainingSet, 0);
        FG = new Instances(TrainingSet, 0);
        DR = new Instances(TrainingSet, 0);
        FR = new Instances(TrainingSet, 0);

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
        if (!userDefinedBags)
            k_NumBags =2* bags;
        log.info("bags = " + k_NumBags);

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
                double results[] = m_Boost.get(sampleNum).distributionForInstance(ins);
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

    private double [] statistics() throws Exception {


        double tp_non_protected = 0;
        double tn_non_protected = 0;
        double tp_protected = 0;
        double tn_protected = 0;
        double fp_non_protected = 0;
        double fn_non_protected = 0;
        double fp_protected = 0;
        double fn_protected = 0;

        for(Instance ins: TrainingSet){
            double label = this.classifyInstance(ins);
            if (ins.stringValue(protectedValueIndex).equals(protectedValueName)) {
                if (label == ins.classValue()) {
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_protected++;
                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        tn_protected++;
                    }
                }else{
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        fn_protected++;
                    }else if (ins.stringValue(ins.classIndex()).equals(otherClass)){
                        fp_protected++;
                    }
                }
            }else{
                if (label == ins.classValue()) {
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_non_protected++;
                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        tn_non_protected++;
                    }
                }else{
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        fn_non_protected++;
                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        fp_non_protected++;
                    }
                }
            }
        }
        double [] output = new double[2];
        output[0]= (tp_non_protected)/(tp_non_protected + fn_non_protected) - (tp_protected)/(tp_protected + fn_protected);
        output[1]= 1 - 0.5*((tp_non_protected + tp_protected)/(tp_non_protected + fn_non_protected + tp_protected + fn_protected) +
                (tn_non_protected + tn_protected)/(tn_non_protected + fp_non_protected + tn_protected + fp_protected));
        return output;
    }

}
