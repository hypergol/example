import tensorflow as tf
from tensorflow.keras import layers
from hypergol import BaseTensorflowModelBlock


def _add_oov(vocabulary):
    return ['', '-OOV-'] + [v for v in vocabulary if v not in ['', '-OOV-']]


class OutputBlock(BaseTensorflowModelBlock):

    def __init__(self, posTypes, **kwargs):
        super(OutputBlock, self).__init__(**kwargs)
        self.posTypes = _add_oov(posTypes)
        self.indexToPosType = tf.constant(
            self.posTypes,
            dtype=tf.string,
            name='indexToPosType'
        )
        self.posTypeToIndex = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.posTypes,
                values=list(range(len(self.posTypes)))
            ),
            default_value=1,
            name='posTypeToIndex'
        )

    def get_pos_type_indices(self, posTypes):
        with tf.name_scope('posTypeIndices'):
            return self.posTypeToIndex.lookup(posTypes)

    def get_pos_types(self, posTypeIndices):
        return tf.gather(params=self.indexToPosType, indices=posTypeIndices, axis=None)
