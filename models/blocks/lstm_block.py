import tensorflow as tf
from tensorflow.keras import layers
from hypergol import BaseTensorflowModelBlock


def get_shape(t):
    return [
        d if s is None else d for s, d
        in zip(t.shape.as_list(), tf.unstack(tf.shape(t)))
    ]


class LstmBlock(BaseTensorflowModelBlock):

    def __init__(self, embeddingDimension, layerCount, posTypeCount, dropoutRate, **kwargs):
        super(LstmBlock, self).__init__(**kwargs)
        self.embeddingDimension = embeddingDimension
        self.dropoutRate = dropoutRate
        self.layerCount = layerCount
        self.posTypeCount = posTypeCount
        self.lstmLayers = []
        for _ in range(self.layerCount):
            self.lstmLayers.append(layers.Bidirectional(
                layer=layers.LSTM(
                    units=self.embeddingDimension,
                    return_sequences=True
                )))
        self.dropoutLayer = layers.Dropout(rate=self.dropoutRate, name='dropoutLayer')
        self.posTypeLogits = layers.Dense(units=self.posTypeCount, name='posTypeLogits')

    def get_pos_type_logits(self, sentenceTokenEmbeddings, sentenceLengths, training):
        values = sentenceTokenEmbeddings
        mask = tf.sequence_mask(lengths=sentenceLengths, maxlen=get_shape(values)[1])
        for lstmLayer in self.lstmLayers:
            values = lstmLayer(values, mask=mask)
        values = self.dropoutLayer(values, training=training)
        return self.posTypeLogits(values)
