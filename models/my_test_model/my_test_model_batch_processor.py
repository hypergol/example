import tensorflow as tf
from hypergol import BaseBatchProcessor

from data_models.evaluation_output import EvaluationOutput
from data_models.model_output import ModelOutput


class MyTestModelBatchProcessor(BaseBatchProcessor):

    def __init__(self, inputDataset, inputBatchSize, outputDataset, maxTokenCount):
        super(MyTestModelBatchProcessor, self).__init__(inputDataset, inputBatchSize, outputDataset)
        self.maxTokenCount = maxTokenCount

    def process_input_batch(self, batch):
        lemmas = []
        ids = []
        sentenceLengths = []
        for sentence in batch:
            ids.append(sentence.get_hash_id())
            lemmas.append([token.lemma for token in sentence.tokens])
            sentenceLengths.append(len(sentence.tokens))
        return {
            'ids': ids,
            'sentenceTokens': tf.ragged.constant(lemmas, dtype=tf.string).to_tensor()[:, :self.maxTokenCount],
            'sentenceLengths': tf.constant(sentenceLengths, dtype=tf.int32),
        }

    def process_training_batch(self, batch):
        lemmas = []
        ids = []
        sentenceLengths = []
        posLabels = []
        for sentence in batch:
            ids.append(sentence.get_hash_id())
            lemmas.append([token.lemma for token in sentence.tokens])
            sentenceLengths.append(len(sentence.tokens))
            posLabels.append([token.posFineType for token in sentence.tokens])
        inputs = {
            'ids': ids,
            'sentenceTokens': tf.ragged.constant(lemmas, dtype=tf.string).to_tensor()[:, :self.maxTokenCount],
            'sentenceLengths': tf.constant(sentenceLengths, dtype=tf.int32),
        }
        targets = tf.ragged.constant(posLabels, dtype=tf.string).to_tensor()[:, :self.maxTokenCount]
        return inputs, targets

    def process_output_batch(self, outputs):
        outputBatch = []
        for o in outputs:
            outputBatch.append(ModelOutput(
                articleId=0,
                sentenceId=0,
                posTags=[pos.decode('utf-8') for pos in o.numpy()]
            ))
        return outputBatch

    def process_evaluation_batch(self, inputs, targets, outputs):
        outputBatch = []
        for id_, i, t, o in zip(inputs['ids'], inputs['sentenceTokens'], targets, outputs):
            outputBatch.append(EvaluationOutput(
                articleId=id_[0],
                sentenceId=id_[1],
                inputs=i.numpy(),
                targets=t,
                outputs=o))
        return outputBatch
