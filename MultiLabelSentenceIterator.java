package org.deeplearning4j.examples.convolution.sentenceclassification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.LabelAwareConverter;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.documentiterator.LabelAwareDocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

public class MultiLabelSentenceIterator implements DataSetIterator{
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public enum UnknownWordHandling {
        RemoveWord, UseUnknownVector
    }

    /**
     * Format of features:<br>
     * CNN1D: For use with 1d convolution layers: Shape [minibatch, vectorSize, sentenceLength]<br>
     * CNN2D: For use with 2d convolution layers: Shape [minibatch, 1, vectorSize, sentenceLength] or [minibatch, 1, sentenceLength, vectorSize],
     * depending on the setting for 'sentencesAlongHeight' configuration.
     */
    public enum Format {
        RNN, CNN1D, CNN2D
    }

    private static final String UNKNOWN_WORD_SENTINEL = "UNKNOWN_WORD_SENTINEL";

    private Format format;
    private LabeledSentenceProvider sentenceProvider;
    private WordVectors wordVectors;
    private TokenizerFactory tokenizerFactory;
    private UnknownWordHandling unknownWordHandling;
    private boolean useNormalizedWordVectors;
    private int minibatchSize;
    private int maxSentenceLength;
    private boolean sentencesAlongHeight;
    private DataSetPreProcessor dataSetPreProcessor;

    private int wordVectorSize;
    public int numClasses;
    private Map<String, Integer> labelClassMap;
    private Map<String, Integer> actualLabelClassMap;
    private INDArray unknown;

    private int cursor = 0;

    private Pair<List<String>, String> preLoadedTokens;

    protected MultiLabelSentenceIterator(Builder builder) {
        this.format = builder.format;
        this.sentenceProvider = builder.sentenceProvider;
        this.wordVectors = builder.wordVectors;
        this.tokenizerFactory = builder.tokenizerFactory;
        this.unknownWordHandling = builder.unknownWordHandling;
        this.useNormalizedWordVectors = builder.useNormalizedWordVectors;
        this.minibatchSize = builder.minibatchSize;
        this.maxSentenceLength = builder.maxSentenceLength;
        this.sentencesAlongHeight = builder.sentencesAlongHeight;
        this.dataSetPreProcessor = builder.dataSetPreProcessor;


        //this.numClasses = this.sentenceProvider.numLabelClasses();
        //2020-3-9 bw
        this.numClasses = this.actualNumberOfLabels(this.sentenceProvider.allLabels());
        
        this.labelClassMap = new HashMap<>();
        this.actualLabelClassMap = new HashMap<>();
        
        int count = 0;
        //First: sort the labels to ensure the same label assignment order (say train vs. test)
        // bw
        List<String> sortedLabels = new ArrayList<>(this.sentenceProvider.allLabels());//包括multi label的
        List<String> actualSortedLabels = new ArrayList<>(this.actualLabels(this.sentenceProvider.allLabels()));//去掉multi label的
        
        Collections.sort(sortedLabels);
        Collections.sort(actualSortedLabels);
        
        this.wordVectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

        for (String s : sortedLabels) {
            this.labelClassMap.put(s, count++);
        }
        count=0;
        for (String s : actualSortedLabels) {
            this.actualLabelClassMap.put(s, count++);
        }
        System.out.println("----------------------------------  original map size : "+labelClassMap.size());
        System.out.println("----------------------------------  actual map size : "+actualLabelClassMap.size());
        
        if (unknownWordHandling == UnknownWordHandling.UseUnknownVector) {
            if (useNormalizedWordVectors) {
                unknown = wordVectors.getWordVectorMatrixNormalized(wordVectors.getUNK());
            } else {
                unknown = wordVectors.getWordVectorMatrix(wordVectors.getUNK());
            }

            if(unknown == null){
                unknown = wordVectors.getWordVectorMatrix(wordVectors.vocab().wordAtIndex(0)).like();
            }
        }
    }

    //2020-3-9 bw 计算时间的标签数量，除去multi label
    public int actualNumberOfLabels(List<String> list)
    {
    	int count=0;
    	for(String str : list)
    	{
    		if(str.contains(","))
    			continue;
    		count++;
    	}
    	return count;
    }
    
    public List<String> actualLabels(List<String> list)
    {
    	List<String> temp = new ArrayList<String>();
    	for(String str : list)
    	{
    		if(str.contains(","))
    			continue;
    		temp.add(str);
    	}
    	return temp;
    }
    
    
    /**
     * Generally used post training time to load a single sentence for predictions
     */
    public INDArray loadSingleSentence(String sentence) {
        List<String> tokens = tokenizeSentence(sentence);
        if(tokens.isEmpty())
            throw new IllegalStateException("No tokens available for input sentence - empty string or no words in vocabulary with RemoveWord unknown handling? Sentence = \"" +
                    sentence + "\"");
        if(format == Format.CNN1D || format == Format.RNN){
            int[] featuresShape = new int[] {1, wordVectorSize, Math.min(maxSentenceLength, tokens.size())};
            INDArray features = Nd4j.create(featuresShape, (format == Format.CNN1D ? 'c' : 'f'));
            INDArrayIndex[] indices = new INDArrayIndex[3];
            indices[0] = NDArrayIndex.point(0);
            for (int i = 0; i < featuresShape[2]; i++) {
                INDArray vector = getVector(tokens.get(i));
                indices[1] = NDArrayIndex.all();
                indices[2] = NDArrayIndex.point(i);
                features.put(indices, vector);
            }
            return features;
        } else {
            int[] featuresShape = new int[] {1, 1, 0, 0};
            if (sentencesAlongHeight) {
                featuresShape[2] = Math.min(maxSentenceLength, tokens.size());
                featuresShape[3] = wordVectorSize;
            } else {
                featuresShape[2] = wordVectorSize;
                featuresShape[3] = Math.min(maxSentenceLength, tokens.size());
            }

            INDArray features = Nd4j.create(featuresShape);
            int length = (sentencesAlongHeight ? featuresShape[2] : featuresShape[3]);
            INDArrayIndex[] indices = new INDArrayIndex[4];
            indices[0] = NDArrayIndex.point(0);
            indices[1] = NDArrayIndex.point(0);
            for (int i = 0; i < length; i++) {
                INDArray vector = getVector(tokens.get(i));

                if (sentencesAlongHeight) {
                    indices[2] = NDArrayIndex.point(i);
                    indices[3] = NDArrayIndex.all();
                } else {
                    indices[2] = NDArrayIndex.all();
                    indices[3] = NDArrayIndex.point(i);
                }

                features.put(indices, vector);
            }

            return features;
        }
    }

    private INDArray getVector(String word) {
        INDArray vector;
        if (unknownWordHandling == UnknownWordHandling.UseUnknownVector && word == UNKNOWN_WORD_SENTINEL) { //Yes, this *should* be using == for the sentinel String here
            vector = unknown;
        } else {
            if (useNormalizedWordVectors) {
                vector = wordVectors.getWordVectorMatrixNormalized(word);
            } else {
                vector = wordVectors.getWordVectorMatrix(word);
            }
        }
        return vector;
    }

    private List<String> tokenizeSentence(String sentence) {
        Tokenizer t = tokenizerFactory.create(sentence);

        List<String> tokens = new ArrayList<>();
        while (t.hasMoreTokens()) {
            String token = t.nextToken();
            if (!wordVectors.outOfVocabularySupported() && !wordVectors.hasWord(token)) {
                switch (unknownWordHandling) {
                    case RemoveWord:
                        continue;
                    case UseUnknownVector:
                        token = UNKNOWN_WORD_SENTINEL;
                }
            }
            tokens.add(token);
        }
        return tokens;
    }

    public Map<String, Integer> getLabelClassMap() {
        return new HashMap<>(labelClassMap);
    }

    @Override
    public List<String> getLabels() {
        //We don't want to just return the list from the LabelledSentenceProvider, as we sorted them earlier to do the
        // String -> Integer mapping
        String[] str = new String[labelClassMap.size()];
        for (Map.Entry<String, Integer> e : labelClassMap.entrySet()) {
            str[e.getValue()] = e.getKey();
        }
        return Arrays.asList(str);
    }

    @Override
    public boolean hasNext() {
        if (sentenceProvider == null) {
            throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
        }

        while (preLoadedTokens == null && sentenceProvider.hasNext()) {
            //Pre-load tokens. Because we filter out empty strings, or sentences with no valid words
            //we need to pre-load some tokens. Otherwise, sentenceProvider could have 1 (invalid) sentence
            //next, hasNext() would return true, but next(int) wouldn't be able to return anything
            preLoadTokens();
        }

        return preLoadedTokens != null;
    }

    private void preLoadTokens() {
        if (preLoadedTokens != null) {
            return;
        }
        Pair<String, String> p = sentenceProvider.nextSentence();
        List<String> tokens = tokenizeSentence(p.getFirst());
        if (!tokens.isEmpty()) {
            preLoadedTokens = new Pair<>(tokens, p.getSecond());
        }
    }

    @Override
    public DataSet next() {
        return next(minibatchSize);
    }

    @Override
    public DataSet next(int num) {
        if (sentenceProvider == null) {
            throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
        }
        if (!hasNext()) {
            throw new NoSuchElementException("No next element");
        }


        List<Pair<List<String>, String>> tokenizedSentences = new ArrayList<>(num);
        int maxLength = -1;
        int minLength = Integer.MAX_VALUE; //Track to we know if we can skip mask creation for "all same length" case
        if (preLoadedTokens != null) {
            tokenizedSentences.add(preLoadedTokens);
            maxLength = Math.max(maxLength, preLoadedTokens.getFirst().size());
            minLength = Math.min(minLength, preLoadedTokens.getFirst().size());
            preLoadedTokens = null;
        }
        for (int i = tokenizedSentences.size(); i < num && sentenceProvider.hasNext(); i++) {
            Pair<String, String> p = sentenceProvider.nextSentence();
            List<String> tokens = tokenizeSentence(p.getFirst());

            if (!tokens.isEmpty()) {
                //Handle edge case: no tokens from sentence
                maxLength = Math.max(maxLength, tokens.size());
                minLength = Math.min(minLength, tokens.size());
                tokenizedSentences.add(new Pair<>(tokens, p.getSecond()));
            } else {
                //Skip the current iterator
                i--;
            }
        }

        if (maxSentenceLength > 0 && maxLength > maxSentenceLength) {
            maxLength = maxSentenceLength;
        }

        int currMinibatchSize = tokenizedSentences.size();
        INDArray labels = Nd4j.create(currMinibatchSize, numClasses);
        for (int i = 0; i < tokenizedSentences.size(); i++) {
            String labelStr = tokenizedSentences.get(i).getSecond();
            if (!labelClassMap.containsKey(labelStr)) {
                throw new IllegalStateException("Got label \"" + labelStr
                                + "\" that is not present in list of LabeledSentenceProvider labels");
            }

			/*
			 * int labelIdx = labelClassMap.get(labelStr); 
			 * labels.putScalar(i, labelIdx,1.0);
			 */
            //2020-3-9 加入对multi label支持,多标签的意图格式为 ：mode_ac,eco_off
            //System.out.println("label string : "+labelStr);
            for(String label : labelStr.split(","))
            {
            	int labelIdx = actualLabelClassMap.get(label);
                labels.putScalar(i, labelIdx, 1.0);
            }
        }

        INDArray features;
        INDArray featuresMask = null;
        if(format == Format.CNN1D || format == Format.RNN){
            int[] featuresShape = new int[]{currMinibatchSize, wordVectorSize, maxLength};
            features = Nd4j.create(featuresShape, (format == Format.CNN1D ? 'c' : 'f'));

            INDArrayIndex[] idxs = new INDArrayIndex[3];
            idxs[1] = NDArrayIndex.all();
            for (int i = 0; i < currMinibatchSize; i++) {
                idxs[0] = NDArrayIndex.point(i);
                List<String> currSentence = tokenizedSentences.get(i).getFirst();
                for (int j = 0; j < currSentence.size() && j < maxSentenceLength; j++) {
                    idxs[2] = NDArrayIndex.point(j);
                    INDArray vector = getVector(currSentence.get(j));
                    features.put(idxs, vector);
                }
            }

            if (minLength != maxLength) {
                featuresMask = Nd4j.create(currMinibatchSize, maxLength);
                for (int i = 0; i < currMinibatchSize; i++) {
                    int sentenceLength = tokenizedSentences.get(i).getFirst().size();
                    if (sentenceLength >= maxLength) {
                        featuresMask.getRow(i).assign(1.0);
                    } else {
                        featuresMask.get(NDArrayIndex.point(i), NDArrayIndex.interval(0, sentenceLength)).assign(1.0);
                    }
                }
            }

        } else {
            int[] featuresShape = new int[4];
            featuresShape[0] = currMinibatchSize;
            featuresShape[1] = 1;
            if (sentencesAlongHeight) {
                featuresShape[2] = maxLength;
                featuresShape[3] = wordVectorSize;
            } else {
                featuresShape[2] = wordVectorSize;
                featuresShape[3] = maxLength;
            }

            features = Nd4j.create(featuresShape);
            INDArrayIndex[] indices = new INDArrayIndex[4];
            indices[1] = NDArrayIndex.point(0);
            for (int i = 0; i < currMinibatchSize; i++) {
                indices[0] = NDArrayIndex.point(i);
                List<String> currSentence = tokenizedSentences.get(i).getFirst();
                for (int j = 0; j < currSentence.size() && j < maxSentenceLength; j++) {
                    INDArray vector = getVector(currSentence.get(j));

                    if (sentencesAlongHeight) {
                        indices[2] = NDArrayIndex.point(j);
                        indices[3] = NDArrayIndex.all();
                    } else {
                        indices[2] = NDArrayIndex.all();
                        indices[3] = NDArrayIndex.point(j);
                    }

                    features.put(indices, vector);
                }
            }

            if (minLength != maxLength) {
                if(sentencesAlongHeight){
                    featuresMask = Nd4j.create(currMinibatchSize, 1, maxLength, 1);
                    for (int i = 0; i < currMinibatchSize; i++) {
                        int sentenceLength = tokenizedSentences.get(i).getFirst().size();
                        if (sentenceLength >= maxLength) {
                            featuresMask.slice(i).assign(1.0);
                        } else {
                            featuresMask.get(NDArrayIndex.point(i), NDArrayIndex.point(0), NDArrayIndex.interval(0, sentenceLength), NDArrayIndex.point(0)).assign(1.0);
                        }
                    }
                } else {
                    featuresMask = Nd4j.create(currMinibatchSize, 1, 1, maxLength);
                    for (int i = 0; i < currMinibatchSize; i++) {
                        int sentenceLength = tokenizedSentences.get(i).getFirst().size();
                        if (sentenceLength >= maxLength) {
                            featuresMask.slice(i).assign(1.0);
                        } else {
                            featuresMask.get(NDArrayIndex.point(i), NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.interval(0, sentenceLength)).assign(1.0);
                        }
                    }
                }
            }
        }

        DataSet ds = new DataSet(features, labels, featuresMask, null);

        if (dataSetPreProcessor != null) {
            dataSetPreProcessor.preProcess(ds);
        }

        cursor += ds.numExamples();
        return ds;
    }

    @Override
    public int inputColumns() {
        return wordVectorSize;
    }

    @Override
    public int totalOutcomes() {
        return numClasses;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
        sentenceProvider.reset();
    }

    @Override
    public int batch() {
        return minibatchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return dataSetPreProcessor;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }

    public static class Builder {

        private Format format;
        private LabeledSentenceProvider sentenceProvider = null;
        private WordVectors wordVectors;
        private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        private UnknownWordHandling unknownWordHandling = UnknownWordHandling.RemoveWord;
        private boolean useNormalizedWordVectors = true;
        private int maxSentenceLength = -1;
        private int minibatchSize = 32;
        private boolean sentencesAlongHeight = true;
        private DataSetPreProcessor dataSetPreProcessor;

        /**
         * @deprecated Due to old default, that will be changed in the future. Use {@link #Builder(Format)} to specify
         * the {@link Format} of the activations
         */
        @Deprecated
        public Builder(){
            //Default for backward compatibility
            this(Format.CNN2D);
        }

        /**
         * @param format The format to use for the features - i.e., for 1D or 2D CNNs
         */
        public Builder( Format format){
            this.format = format;
        }

        /**
         * Specify how the (labelled) sentences / documents should be provided
         */
        public Builder sentenceProvider(LabeledSentenceProvider labeledSentenceProvider) {
            this.sentenceProvider = labeledSentenceProvider;
            return this;
        }

        /**
         * Specify how the (labelled) sentences / documents should be provided
         */
        public Builder sentenceProvider(LabelAwareIterator iterator,  List<String> labels) {
            LabelAwareConverter converter = new LabelAwareConverter(iterator, labels);
            return sentenceProvider(converter);
        }

        /**
         * Specify how the (labelled) sentences / documents should be provided
         */
        public Builder sentenceProvider(LabelAwareDocumentIterator iterator,  List<String> labels) {
            DocumentIteratorConverter converter = new DocumentIteratorConverter(iterator);
            return sentenceProvider(converter, labels);
        }

        /**
         * Specify how the (labelled) sentences / documents should be provided
         */
        public Builder sentenceProvider(LabelAwareSentenceIterator iterator,  List<String> labels) {
            SentenceIteratorConverter converter = new SentenceIteratorConverter(iterator);
            return sentenceProvider(converter, labels);
        }


        /**
         * Provide the WordVectors instance that should be used for training
         */
        public Builder wordVectors(WordVectors wordVectors) {
            this.wordVectors = wordVectors;
            return this;
        }

        /**
         * The {@link TokenizerFactory} that should be used. Defaults to {@link DefaultTokenizerFactory}
         */
        public Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        /**
         * Specify how unknown words (those that don't have a word vector in the provided WordVectors instance) should be
         * handled. Default: remove/ignore unknown words.
         */
        public Builder unknownWordHandling(UnknownWordHandling unknownWordHandling) {
            this.unknownWordHandling = unknownWordHandling;
            return this;
        }

        /**
         * Minibatch size to use for the DataSetIterator
         */
        public Builder minibatchSize(int minibatchSize) {
            this.minibatchSize = minibatchSize;
            return this;
        }

        /**
         * Whether normalized word vectors should be used. Default: true
         */
        public Builder useNormalizedWordVectors(boolean useNormalizedWordVectors) {
            this.useNormalizedWordVectors = useNormalizedWordVectors;
            return this;
        }

        /**
         * Maximum sentence/document length. If sentences exceed this, they will be truncated to this length by
         * taking the first 'maxSentenceLength' known words.
         */
        public Builder maxSentenceLength(int maxSentenceLength) {
            this.maxSentenceLength = maxSentenceLength;
            return this;
        }

        /**
         * If true (default): output features data with shape [minibatchSize, 1, maxSentenceLength, wordVectorSize]<br>
         * If false: output features with shape [minibatchSize, 1, wordVectorSize, maxSentenceLength]
         */
        public Builder sentencesAlongHeight(boolean sentencesAlongHeight) {
            this.sentencesAlongHeight = sentencesAlongHeight;
            return this;
        }

        /**
         * Optional DataSetPreProcessor
         */
        public Builder dataSetPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
            this.dataSetPreProcessor = dataSetPreProcessor;
            return this;
        }

        public MultiLabelSentenceIterator build() {
            if (wordVectors == null) {
                throw new IllegalStateException(
                                "Cannot build CnnSentenceDataSetIterator without a WordVectors instance");
            }

            return new MultiLabelSentenceIterator(this);
        }

    }
}
