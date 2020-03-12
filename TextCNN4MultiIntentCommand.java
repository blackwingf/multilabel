/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.convolution.sentenceclassification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.examples.convolution.sentenceclassification.MultiLabelSentenceIterator.Format;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMultiLabel;

/**
 * Convolutional Neural Networks for Sentence Classification - https://arxiv.org/abs/1408.5882
 *
 * Specifically, this is the 'static' model from there
 * 用户作为空调命令词分类，多意图版本
 * 2020-3-9
 * @author Alex Black
 */
public class TextCNN4MultiIntentCommand {

    /** Location to save and extract the training/testing data */
    public static final String DATA_PATH = "d:/corpus/char";
    static String modelPath = "e:/models/cmd_char";
    /** Location (local file system) for the Google News vectors. Set this manually. */
    //private static final String WORD_VECTORS_PATH = "/PATH/TO/YOUR/VECTORS/GoogleNews-vectors-negative300.bin.gz";
    private static final String WORD_VECTORS_PATH = "e:/models/w2v_all_char";//w2v_all_word

    public static void main(String[] args) throws Exception {
        //noinspection ConstantConditions
        if(WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")){
            throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
        }

        //Basic configuration
        int batchSize = 32;
        int vectorSize = 100;               //Size of the word vectors. 300 in the Google News model
        int nEpochs = 3;                    //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345); //For shuffling repeatability

        //Set up the network configuration. Note that we have multiple convolution layers, each wih filter
        //widths of 3, 4 and 5 as per Kim (2014) paper.

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        //Load word vectors and get the DataSetIterators for training and testing
        System.out.println("Loading word vectors and creating DataSetIterators");
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        DataSetIterator trainIter = getDataSetIterator(true, wordVectors, batchSize, truncateReviewsToLength, rng);
        DataSetIterator testIter = getDataSetIterator(false, wordVectors, batchSize, truncateReviewsToLength, rng);
        
        int actualNumberOfLabels = ((MultiLabelSentenceIterator) trainIter).numClasses;
        System.out.println("actualNumberOfLabels = "+actualNumberOfLabels);
        
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(new Adam(0.01))
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .l2(0.0001)
            .graphBuilder()
            .addInputs("input")
            .addLayer("cnn3", new ConvolutionLayer.Builder()
                .kernelSize(3,vectorSize)
                .stride(1,vectorSize)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(4,vectorSize)
                .stride(1,vectorSize)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                .kernelSize(5,vectorSize)
                .stride(1,vectorSize)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            //MergeVertex performs depth concatenation on activations: 3x[minibatch,100,length,300] to 1x[minibatch,300,length,300]
            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
            //Global pooling: pool over x/y locations (dimensions 2 and 3): Activations [minibatch,300,length,300] to [minibatch, 300]
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(globalPoolingType)
                .dropOut(0.5)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(new LossMultiLabel())
                .activation(Activation.SIGMOID)
                .nOut(actualNumberOfLabels)    //意图数量
                .build(), "globalPool")
            .setOutputs("out")
            //Input has shape [minibatch, channels=1, length=1 to 256, 300]
            .setInputTypes(InputType.convolutional(truncateReviewsToLength, vectorSize, 1))
            .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("Number of parameters by layer:");
        for(Layer l : net.getLayers() ){
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }


        System.out.println("Starting training");
        net.setListeners(new ScoreIterationListener(100), new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
        net.fit(trainIter, nEpochs);

        // bw 保存model
        ModelSerializer.writeModel(net, modelPath,true);
        System.out.println("----------------------------- finish training cmd model -----------------------------");
        
        //After training: load a single sentence and generate a prediction
        //String pathFirstNegativeFile = FilenameUtils.concat(DATA_PATH, "test.txt");
        //String contentsFirstNegative = FileUtils.readFileToString(new File(pathFirstNegativeFile), (Charset.forName("utf-8")));
		
		//String corpus = TextCNN4Command.readCorpus(DATA_PATH + File.separator + "test.txt");
		//String[] datas = corpus.split("\\s+");
        String corpus = "开 一 下 空 调 吧";
		INDArray featuresFirstNegative = ((MultiLabelSentenceIterator) testIter).loadSingleSentence(corpus);

		INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);
		List<String> labels = ((MultiLabelSentenceIterator) testIter).actualLabels(testIter.getLabels());
		Collections.sort(labels);
		
		System.out.println("\n\nPredictions for cmd :" + corpus);
		for (int i = 0; i < labels.size(); i++) {
			System.out.println(i+"	P(" + labels.get(i) + ") = " + predictionsFirstNegative.getDouble(i));
		}
    }

    public static String readCorpus(String path)
    {
    	String corpus="";
    	try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(path)),"utf-8"));
			corpus = br.readLine();
		} catch (Exception e) {
			e.printStackTrace();
		}
    	return corpus;
    }

    private static DataSetIterator getDataSetIterator(boolean isTraining, WordVectors wordVectors, int minibatchSize,
                                                      int maxSentenceLength, Random rng ){
        String path = FilenameUtils.concat(DATA_PATH, (isTraining ? "train/" : "test/"));
        Map<String,List<File>> reviewFilesMap = new HashMap<>();
        
        LabeledSentenceProvider sentenceProvider = new FileLabeledSentenceProvider(prepareCorpus(reviewFilesMap,path), rng);

        return new MultiLabelSentenceIterator.Builder(Format.CNN2D)
            .sentenceProvider(sentenceProvider)
            .wordVectors(wordVectors)
            .minibatchSize(minibatchSize)
            .maxSentenceLength(maxSentenceLength)
            .useNormalizedWordVectors(false)
            .build();
    }
    
    private static Map<String,List<File>> prepareCorpus(Map<String,List<File>> reviewFilesMap, String path){
    	File dir = new File(path);
    	// list all directories
    	File [] dirs = dir.listFiles(new FileFilter() {
			@Override
			public boolean accept(File f) {
				boolean isDir = f.isDirectory();
				return isDir;
			}
    	});
    	
    	for(File f : dirs)
    	{
    		//System.out.println("dir name : "+f.getName()+"     files count : "+f.listFiles().length);
    		reviewFilesMap.put(f.getName(), Arrays.asList(Objects.requireNonNull(f.listFiles())));
    	}
    	
    	return reviewFilesMap;
    }
    
}
