package demo;
/*
 *2016年12月22日	下午3:12:50
 *@Author Pin-Wang
 *@E-mail 1228935432@qq.com
*/

import java.io.File;
import redis.clients.jedis.Jedis;

import weka.classifiers.*;
import weka.core.*;
import weka.core.converters.ArffLoader;

public class Main {
	public static void main(String[] args) {
		
		 Jedis jedis = new Jedis("101.200.56.4");
		 System.out.println("Server is running: "+jedis.ping());
		 System.out.println("已经连接阿里云redis");
		// jedis.pub

	       try {
	          
	           Classifier classifier1;
	           Classifier classifier2;
	           Classifier classifier3;
	           Classifier classifier4;
	 
	          
	           File inputFile = new File("C:\\Users\\Pin-Wang\\Desktop\\weka数据\\bank-data-final.arff");// 训练语料文件
	           ArffLoader atf = new ArffLoader();
	           atf.setFile(inputFile);
	           Instances instancesTrain = atf.getDataSet(); // 读入训练文件
	           
	           int numInstance=instancesTrain.numInstances();
	           System.out.println("训练文件有：" +numInstance+"行");
	           
	           int numAttribute=instancesTrain.numAttributes();
	           System.out.println("训练文件有：" +numAttribute+"列");
	          
	           inputFile = new File("C:\\Users\\Pin-Wang\\Desktop\\weka数据\\bank-data-final.arff");// 测试语料文件
	           atf.setFile(inputFile);
	           Instances instancesTest = atf.getDataSet(); // 读入测试文件
	 
	          
	           //instancesTest.setClassIndex(0);
	           instancesTrain.setClassIndex(numAttribute-1);
	 
	          
	           // 朴素贝叶斯算法
	           classifier1 = (Classifier) Class.forName(
	                  "weka.classifiers.bayes.NaiveBayes").newInstance();
	           // 决策树
	           classifier2 = (Classifier) Class.forName(
	                  "weka.classifiers.trees.J48").newInstance();
	           // Zero
	           classifier3 = (Classifier) Class.forName(
	                  "weka.classifiers.rules.ZeroR").newInstance();
	           // LibSVM
	           //classifier4 = (Classifier) Class.forName(
	            //      "weka.classifiers.functions.LibSVM").newInstance();
	 
	          
	          // classifier4.buildClassifier(instancesTrain);
	           classifier1.buildClassifier(instancesTrain);
	           classifier2.buildClassifier(instancesTrain);
	           classifier3.buildClassifier(instancesTrain);
	           int erro=0;
	           StringBuilder ids=new StringBuilder();
	           for(int i=0;i<numInstance;i++){
	        	   if(classifier2.classifyInstance(instancesTrain.instance(i))==0.0){
	        		   String[] arr=instancesTrain.instance(i).toString().split(",");
	        		   ids.append(arr[0]+",");
	        		   erro++;
	        	   }
	           }
	           String result=ids.substring(0,ids.length()-1).toString();
	           //System.out.println(result);
	          jedis.set("pep_ids", result);
	           
	           // System.out.println("错误率"+(double)erro/numInstance);
	
	          
	           Evaluation eval = new Evaluation(instancesTrain);
	          
	           //  eval.evaluateModel(classifier4, instancesTest);
	          // System.out.println(eval.errorRate());
	           eval.evaluateModel(classifier2, instancesTrain);
	          System.out.println(eval.errorRate());
	           
	           /*
	           eval.evaluateModel(classifier1, instancesTest);
	           System.out.println(eval.errorRate());
	       
	           eval.evaluateModel(classifier3, instancesTest);
	           System.out.println(eval.errorRate());
	           */
	          
	       } catch (Exception e) {
	           e.printStackTrace(); 
	       }
	    }
	
	/*
	//如果只有训练集，采用十交叉验证的方法，将上面的第5步和第6步更改为如下代码：
	          
	           Evaluation eval = new Evaluation(instancesTrain);
	           eval.crossValidateModel(classifier4, instancesTrain, 10, new Random(1));
	           System.out.println(eval.errorRate());
	           eval.crossValidateModel(classifier1, instancesTrain, 10, new Random(1));
	           System.out.println(eval.errorRate());
	           eval.crossValidateModel(classifier2, instancesTrain, 10, new Random(1));
	           System.out.println(eval.errorRate());
	           eval.crossValidateModel(classifier3, instancesTrain, 10, new Random(1));
	           System.out.println(eval.errorRate());
//	如果需要保存和加载分类器模型参数，在第5步和第6步之间加入如下代码：
	          
	           SerializationHelper.write("LibSVM.model", classifier4);
	           SerializationHelper.write("NaiveBayes.model", classifier1);
	           SerializationHelper.write("J48.model", classifier2);
	           SerializationHelper.write("ZeroR.model", classifier3);
	          
	          
	           Classifier classifier8 = (Classifier) weka.core.SerializationHelper.read("LibSVM.model");
	           Classifier classifier5 = (Classifier) weka.core.SerializationHelper.read("NaiveBayes.model");
	           Classifier classifier6 = (Classifier) weka.core.SerializationHelper.read("J48.model");
	           Classifier classifier7 = (Classifier) weka.core.SerializationHelper.read("ZeroR.model");
*/
}
