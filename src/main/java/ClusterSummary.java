import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by iosifidis on 25.01.19.
 */
public class ClusterSummary {

    private double totalInstances;

    public double getBagInstances() {
        return bagInstances;
    }

    public void setBagInstances(double bagInstances) {
        this.bagInstances = bagInstances;
    }

    private double bagInstances;

    private ArrayList<Integer> arrayOfClusterSize;
    private ArrayList<Instances> arrayOfCluster;

    public ClusterSummary() {
        arrayOfCluster = new ArrayList<Instances>();
        arrayOfClusterSize = new ArrayList<Integer>();
    }

    public void setTotalInstances(double totalInstances) {
        this.totalInstances = totalInstances;
    }

    public void insertCluster(Instances instances, double size) {
        arrayOfCluster.add(instances);

        arrayOfClusterSize.add((int)(bagInstances*(size/totalInstances)));
    }

    public Instances getPoints() {
        Instances output = new Instances(arrayOfCluster.get(0),0);
//        System.out.println("Clusters = " + arrayOfCluster.size());

        for (int i=0; i< arrayOfCluster.size(); i++){
            final Random rand = new Random((int) System.currentTimeMillis());   // create seeded number generator
//            System.out.println("Cluster's " + i + " size = " + arrayOfClusterSize.get(i));

            arrayOfCluster.get(i).randomize(rand);
            try {
                for (int j = 0; j < arrayOfClusterSize.get(i) - 1; j++) {
//                    System.out.println(j);
                    output.add(arrayOfCluster.get(i).get(j));
                }
            } catch (Exception e){
                continue;
            }
//            System.out.println("withdraw = " + output.size());
        }

        return output;
    }
}
