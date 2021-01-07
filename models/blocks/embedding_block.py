import tensorflow as tf
from tensorflow.keras import layers
from hypergol import BaseTensorflowModelBlock


def _add_oov(vocabulary):
    return ['', '-OOV-'] + [v for v in vocabulary if v not in ['', '-OOV-']]


class EmbeddingBlock(BaseTensorflowModelBlock):

    def __init__(self, vocabulary, embeddingDimension, **kwargs):
        super(EmbeddingBlock, self).__init__(**kwargs)
        self.vocabulary = _add_oov(vocabulary)
        self.embeddingDimension = embeddingDimension
        self.tokenToIndex = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.vocabulary,
                values=list(range(len(self.vocabulary)))
            ),
            default_value=1,
            name='tokenToIndex'
        )
        self.embeddingMatrix = tf.keras.layers.Embedding(
            input_dim=len(self.vocabulary),
            output_dim=self.embeddingDimension,
            embeddings_initializer=tf.random_normal_initializer(),
            name='embeddingMatrix'
        )

    def get_sentence_token_embeddings(self, sentenceTokens):
        with tf.name_scope('sentenceTokenIndices'):
            sentenceTokenIndices = self.tokenToIndex.lookup(sentenceTokens)
        with tf.name_scope('sentenceTokenEmbeddings'):
            return self.embeddingMatrix(inputs=sentenceTokenIndices)
