import tensorflow as tf
from hypergol import BaseTensorflowModel


def get_pr_f1(predictions, truths):
    truePositives = tf.reduce_sum(tf.keras.backend.cast(tf.logical_and(predictions, truths), dtype=tf.int32))
    falsePositives = tf.reduce_sum(tf.keras.backend.cast(tf.logical_and(predictions, tf.logical_not(truths)), dtype=tf.int32))
    falseNegatives = tf.reduce_sum(tf.keras.backend.cast(tf.logical_and(tf.logical_not(predictions), truths), dtype=tf.int32))
    precision = truePositives / tf.maximum(1, truePositives + falsePositives)
    recall = truePositives / tf.maximum(1, truePositives + falseNegatives)
    f1 = 2 * precision * recall / tf.maximum(10e-16, precision + recall)
    return precision, recall, f1


class MyTestModel(BaseTensorflowModel):

    def __init__(self, embeddingBlock, lstmBlock, outputBlock, **kwargs):
        super(MyTestModel, self).__init__(**kwargs)
        self.embeddingBlock = embeddingBlock
        self.lstmBlock = lstmBlock
        self.outputBlock = outputBlock

    def get_loss(self, targets, training, ids, sentenceTokens, sentenceLengths):
        posTypeLogits = self.get_pos_type_logits(sentenceTokens, sentenceLengths, training=training)
        posTypeIndices = self.outputBlock.get_pos_type_indices(posTypes=targets)
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            y_true=posTypeIndices,
            y_pred=posTypeLogits,
            from_logits=True,
        ))

    def get_pos_type_logits(self, sentenceTokens, sentenceLengths, training):
        sentenceTokenEmbeddings = self.embeddingBlock.get_sentence_token_embeddings(sentenceTokens)
        return self.lstmBlock.get_pos_type_logits(sentenceTokenEmbeddings, sentenceLengths, training)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='ids'),
        tf.TensorSpec(shape=[None, None], dtype=tf.string, name='sentenceTokens'),
        tf.TensorSpec(shape=[None], dtype=tf.int32, name='sentenceLengths')
    ])
    def get_outputs(self, ids, sentenceTokens, sentenceLengths):
        posTypeLogits = self.get_pos_type_logits(sentenceTokens, sentenceLengths, training=False)
        posTypeProbabilites = tf.nn.softmax(logits=posTypeLogits, axis=2)
        posTypeIndices = tf.argmax(input=posTypeProbabilites, axis=2, output_type=tf.int32)
        return self.outputBlock.get_pos_types(posTypeIndices)

    def produce_metrics(self, targets, training, globalStep, ids, sentenceTokens, sentenceLengths):
        predictedPosLabels = self.get_outputs(ids, sentenceTokens, sentenceLengths)
        for posType in self.outputBlock.posTypes[2:]:
            predictions = tf.equal(x=predictedPosLabels, y=posType)
            truths = tf.equal(x=targets, y=posType)
            if tf.reduce_any(truths):
                precision, recall, f1 = get_pr_f1(predictions, truths)
                tf.summary.scalar(name=f'POS_{posType}_Precision', data=precision, step=globalStep)
                tf.summary.scalar(name=f'POS_{posType}_Recall', data=recall, step=globalStep)
                tf.summary.scalar(name=f'POS_{posType}_F1', data=f1, step=globalStep)
        return predictedPosLabels
