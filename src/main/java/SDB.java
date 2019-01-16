/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    CustomAdaBoostM1.java
 *    Copyright (C) 1999-2014 University of Waikato, Hamilton, New Zealand
 *
 */

import org.apache.log4j.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

import static java.lang.Math.abs;
import static java.lang.Math.pow;

public class SDB extends IteratedSingleClassifierEnhancer
        implements Classifier {
  private final static Logger log = Logger.getLogger(SDB.class.getName());


  private String protectedValueName;
  private int protectedValueIndex;
  private String targetClass;
  private String otherClass;
  private double eqOp = 0;
  private double threshold = 0.0;

  /** Array for storing the weights for the votes. */
  protected double[] m_Betas;
  protected double bound;
  protected boolean afterTraining = false;

  /** The number of successfully generated base classifiers. */
  protected int m_NumIterationsPerformed;

  private ArrayList<Double> equalOpp;

  /** The number of classes */
  protected int m_NumClasses;



  /** The (weighted) training data */
  protected Instances m_TrainingData;
  private double sumOfAlhpas = 0;
  private boolean RRBOption = false;
  private double flipProportion;
  private Instances flipedInstances;


  public void setFlipProportion(double flipProportion) {
    this.flipProportion = flipProportion;
  }


  public double getRrb() {
    return this.rrbVaule * 100;
  }

  public void setRrbVaule(double rrbVaule) {
    this.rrbVaule = rrbVaule;
  }

  private double rrbVaule;

  public boolean isRRBOption() {
    return RRBOption;
  }

  public void setRRBOption(boolean RRBOption) {
    this.RRBOption = RRBOption;
  }

  private void TestSetPerformance(int iteration, Instances currentTraininSet) throws Exception {
    final Evaluation eval = new Evaluation(currentTraininSet);
    eval.evaluateModel(this, currentTraininSet);
    double disc = calculateEquallizedOpportunity(this, currentTraininSet);

    double reweight = (1 - abs(disc)) / abs(disc);
    equalOpp.add(pow(Math.log(reweight),2));
    log.info("Iteration = " + iteration + ", Accuracy = " + eval.pctCorrect() + ", au-PRC = " + eval.weightedAreaUnderPRC()*100 + ", au-ROC = " + eval.weightedAreaUnderROC()*100+ ", equallized opportunity = " + disc + ", normalized score = " + equalOpp.get(iteration));
  }

  public SDB(Classifier baseClassifier,
             int protectedValueIndex,
             String protectedValueName,
             String targetClass,
             String otherClass,
             double bound) throws Exception {

    this.bound = bound;
    this.protectedValueIndex = protectedValueIndex;
    this.protectedValueName = protectedValueName;
    this.targetClass = targetClass;
    this.otherClass = otherClass;
    equalOpp = new ArrayList<Double>();
    setClassifier(baseClassifier);
  }


  private double calculateEquallizedOpportunity(Classifier m_classifier, Instances m_trainingData) throws Exception {
    Instances TestingPredictions = new Instances(m_trainingData);

    double tp_male = 0;
    double tn_male = 0;
    double tp_female = 0;
    double tn_female = 0;
    double fp_male = 0;
    double fn_male = 0;
    double fp_female = 0;
    double fn_female = 0;

    for(Instance ins: TestingPredictions){
      double label = m_classifier.classifyInstance(ins);
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

  /**
   * Method used to build the classifier.
   */
  public void buildClassifier(Instances data) throws Exception {

    // Initialize classifier
    initializeClassifier(data);

    // Perform boosting iterations
    while (next()) {};

    // Clean up
    done();

  }

  /**
   * Initialize the classifier.
   *
   * @param data the training data to be used for generating the boosted
   *          classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void initializeClassifier(Instances data) throws Exception {
    super.buildClassifier(data);

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();


    m_NumClasses = data.numClasses();
    m_Betas = new double[m_Classifiers.length];
    m_NumIterationsPerformed = 0;
    m_TrainingData = new Instances(data);
    if(RRBOption){
      addSyntheticBias();
    }

  }

  private void addSyntheticBias() {
    // data are already in random order due to initial shuffling
    int pseudoProtectedCount = 0;
    flipedInstances = new Instances(m_TrainingData, 0);
    ArrayList<Integer> tempIndexes = new ArrayList<Integer>();
    int count = 0;
    for (Instance instance: m_TrainingData){
      if (!instance.stringValue(protectedValueIndex).equals(protectedValueName) && !instance.stringValue(instance.classIndex()).equals(targetClass)){
        tempIndexes.add(count);
        pseudoProtectedCount += 1;
      }
      count +=1;
    }
//    log.info("FG = " + pseudoProtectedCount );
    int m = (int)(pseudoProtectedCount*flipProportion);
    for(int i=0; i< m; i++){
      m_TrainingData.get(tempIndexes.get(i)).setClassValue(targetClass);
      m_TrainingData.get(tempIndexes.get(i)).setClassValue(1.0);
      flipedInstances.add(m_TrainingData.get(tempIndexes.get(i)));
    }
    pseudoProtectedCount = 0;
    for (Instance instance: m_TrainingData){
      if (!instance.stringValue(protectedValueIndex).equals(protectedValueName) && !instance.stringValue(instance.classIndex()).equals(targetClass)){
        tempIndexes.add(count);
        pseudoProtectedCount += 1;
      }
    }

//    log.info("FG = " + pseudoProtectedCount );

  }
  /**
   * Perform the next boosting iteration.
   *
   * @throws Exception if an unforeseen problem occurs
   */
  public boolean next() throws Exception {

    // Have we reached the maximum?
    if (m_NumIterationsPerformed >= m_NumIterations) {
      return false;
    }

    // Select instances to train the classifier on
    Instances trainData = new Instances(m_TrainingData);


    double epsilon = 0;
    m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainData);

    // Evaluate the classifier
    Evaluation evaluation = new Evaluation(m_TrainingData); // Does this need to be a copy
    evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed],
            m_TrainingData);
    epsilon = evaluation.errorRate();

    // Stop if error too big or 0
    if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0)) {
      if (m_NumIterationsPerformed == 0) {
        m_NumIterationsPerformed = 1; // If we're the first we have to use it
      }
      return false;
    }

    // Determine the weight to assign to this model
    double reweight = (1 - epsilon) / epsilon;
    m_Betas[m_NumIterationsPerformed] = Math.log(reweight);

    setWeights(m_TrainingData, reweight);

    m_NumIterationsPerformed++;
    return true;
  }


  private void calculateMajorityThreshold(double signatureFlag) throws Exception {
    Instances FG = new Instances(m_TrainingData, 0);
    Instances DG = new Instances(m_TrainingData, 0);

    for (Instance instance : m_TrainingData){
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

//    ArrayList<Double> probabilitiesMale = new ArrayList<Double>();
    ArrayList<Instance> misclassifiedProtected = new ArrayList<Instance>();
    ArrayList<Double> thitaValues = new ArrayList<Double>();

    for (Instance instance : FG){
      if (this.classifyInstance(instance) == instance.classValue()){
        TPMale +=1;
      }else{
        FNMale +=1;
//        probabilitiesMale.add(this.distributionForInstance(instance)[1]);
      }
    }

    for (Instance instance : DG){
      if (this.classifyInstance(instance) == instance.classValue()){
        TPFemale +=1;
      }else{
        FNFemale +=1;
        misclassifiedProtected.add(instance);
//        probabilitiesFemale.add(this.distributionForInstance(instance)[1]);
      }
    }

    int y = (int) ((TPMale / (malePos)) * femalePos - TPFemale);
    if (y == misclassifiedProtected.size())
      y -= 1;

    for (Instance instance: misclassifiedProtected){
      thitaValues.add(calculateThita(instance));
    }
    Collections.sort(thitaValues);
    Collections.reverse(thitaValues);
    try{
      threshold = thitaValues.get(y);
    }catch (ArrayIndexOutOfBoundsException e){
      threshold= 0;
    }
  }

  private void calculateRBB() throws Exception {
    final Evaluation eval = new Evaluation(m_TrainingData);

    eval.evaluateModel(this, flipedInstances);
    setRrbVaule(eval.errorRate());
  }


  public void done() throws Exception {

    double disc = 100*calculateEquallizedOpportunity(this, m_TrainingData);
    Evaluation firstEval = new Evaluation(m_TrainingData);
    firstEval.evaluateModel(this, m_TrainingData);
//    log.info("Overall Training Set After change: Accuracy = " + firstEval.pctCorrect() + ", au-PRC = " + firstEval.weightedAreaUnderPRC()*100 + ", au-ROC = " + firstEval.weightedAreaUnderROC()*100+ ", equallized opportunity = " + disc );

//    if ((abs(disc) <= bound ))
//      return ;

    for (double alpha : m_Betas){
      sumOfAlhpas += alpha;
    }

    calculateMajorityThreshold(disc);
    afterTraining = true;

    if(RRBOption){
      calculateRBB();
    }
  }

  /**
   * Sets the weights for the next iteration.
   *
   * @param training the training instances
   * @param reweight the reweighting factor
   * @throws Exception if something goes wrong
   */
  protected void setWeights(Instances training, double reweight)
          throws Exception {

    double oldSumOfWeights, newSumOfWeights;

    oldSumOfWeights = training.sumOfWeights();
    Enumeration<Instance> enu = training.enumerateInstances();
    while (enu.hasMoreElements()) {
      Instance instance = enu.nextElement();
      if (!Utils.eq(
              m_Classifiers[m_NumIterationsPerformed].classifyInstance(instance),
              instance.classValue())) {
        instance.setWeight(instance.weight() * reweight);
      }
    }
    // Renormalize weights
    newSumOfWeights = training.sumOfWeights();
    enu = training.enumerateInstances();
    while (enu.hasMoreElements()) {
      Instance instance = enu.nextElement();
      instance.setWeight(instance.weight() * oldSumOfWeights / newSumOfWeights);
    }
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

  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @throws Exception if instance could not be classified successfully
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    double[] sums = new double[instance.numClasses()];
    if (!afterTraining) {
      for (int i = 0; i < m_NumIterationsPerformed; i++) {
        sums[(int) m_Classifiers[i].classifyInstance(instance)] += m_Betas[i];
      }
    } else if (afterTraining) {

      for (int i = 0; i < m_NumIterationsPerformed; i++) {
        sums[(int) m_Classifiers[i].classifyInstance(instance)] += m_Betas[i];
      }

      double value =  convertToRange((sums[1])/sumOfAlhpas);
      if (!instance.stringValue(protectedValueIndex).equals(protectedValueName)) {
        if (value  < 0){
          sums[0] = 1;
          sums[1] = 0;
        }else{
          sums[0] = 0;
          sums[1] = 1;
        }

      }else{
        if (value  < threshold){
          sums[0] = 1;
          sums[1] = 0;
        }else{
          sums[0] = 0;
          sums[1] = 1;
        }

      }
    }
    return Utils.logs2probs(sums);
  }

  private double convertToRange(double x){
    double a = 0.0, b =1.0, c = -1.0, d= 1.0;

    return c +2*(x);
  }

  public double calculateThita(Instance instance) throws Exception {
    double[] sums = new double[instance.numClasses()];
    for (int i = 0; i < m_NumIterationsPerformed; i++) {
      sums[(int) m_Classifiers[i].classifyInstance(instance)] += m_Betas[i];
    }
    return convertToRange(( sums[1])/sumOfAlhpas);
  }

}
