import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    LSTM, Dense, Bidirectional, MultiHeadAttention, GlobalAveragePooling1D, Layer
)
from .modules import AdaptedCorrelation1D, AdaptedTemporalWeighting1D

# Register the custom layer with Keras for serialization
@tf.keras.utils.register_keras_serializable()
class SelfAttentionWrapper(Layer):
    """Wraps MultiHeadAttention for use in Sequential models."""
    def __init__(self, mha_layer, **kwargs):
        super().__init__(**kwargs)
        self.mha = mha_layer
        if not isinstance(self.mha, MultiHeadAttention):
             raise TypeError(f"Expected mha_layer to be an instance of MultiHeadAttention, but got {type(mha_layer)}")

    def build(self, input_shape):
        # Keras handles building the nested MHA layer implicitly.
        super().build(input_shape) # Mark the wrapper layer as built

    def call(self, inputs):
        # Self-attention: inputs are query, value, and key
        return self.mha(query=inputs, value=inputs, key=inputs)

    def compute_output_shape(self, input_shape):
        # Output shape matches input shape for self-attention
        return self.mha.compute_output_shape(query_shape=input_shape, value_shape=input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"mha_layer": tf.keras.layers.serialize(self.mha)})
        return config

    @classmethod
    def from_config(cls, config):
        mha_layer_config = config.pop("mha_layer")
        mha_layer = tf.keras.layers.deserialize(mha_layer_config)
        return cls(mha_layer=mha_layer, **config)


def build_conv1d_lstm_model(input_shape: tuple, num_classes: int, config: dict) -> tf.keras.Model:
    """
    Builds a model with Conv1D, optional CorrNet+-inspired layers, LSTM/BiLSTM,
    and MultiHeadAttention based on the provided configuration.

    Args:
        input_shape: Tuple indicating the input shape (max_seq_length, num_features).
        num_classes: The number of output classes (signs).
        config: The configuration dictionary.

    Returns:
        A compiled Keras Sequential model.
    """
    model_cfg = config.get('model', {})
    use_correlation = model_cfg.get('use_correlation', False)
    use_temporal_weighting = model_cfg.get('use_temporal_weighting', False)
    use_bilstm = model_cfg.get('use_bilstm', True)
    mha_heads = model_cfg.get('mha_heads', 4)
    mha_key_dim = model_cfg.get('mha_key_dim', 128)

    model_layers = [
        Input(shape=input_shape),
        # --- Conv Block 1 ---
        Conv1D(filters=model_cfg.get('conv1_filters', 64), kernel_size=model_cfg.get('conv1_kernel', 3), activation='relu', padding='same', name='conv1'),
        BatchNormalization(name='bn1'),
        MaxPooling1D(pool_size=model_cfg.get('pool1_size', 2), padding='same', name='pool1'),
        Dropout(model_cfg.get('dropout_conv', 0.3), name='drop1'),
        # --- Conv Block 2 ---
        Conv1D(filters=model_cfg.get('conv2_filters', 128), kernel_size=model_cfg.get('conv2_kernel', 3), activation='relu', padding='same', name='conv2'),
        BatchNormalization(name='bn2'),
        Dropout(model_cfg.get('dropout_conv', 0.3), name='drop2'),
    ]

    # --- Optional CorrNet+-inspired Blocks ---
    if use_correlation:
        model_layers.extend([
            AdaptedCorrelation1D(
                neighbors=model_cfg.get('corr_neighbors', 3),
                reduction_factor=model_cfg.get('corr_reduction_factor', 16),
                initial_alpha=model_cfg.get('correlation_alpha', 0.1),
                use_attention=model_cfg.get('corr_use_attention', True),
                name='correlation'
            ),
            BatchNormalization(name='bn_corr'),
            Dropout(model_cfg.get('dropout_corr', 0.4), name='drop_corr')
        ])

    if use_temporal_weighting:
        model_layers.extend([
            AdaptedTemporalWeighting1D(
                 reduction_factor=model_cfg.get('temporal_reduction_factor', 16),
                 num_convs=model_cfg.get('temporal_num_convs', 3),
                 initial_alpha=model_cfg.get('temporal_weighting_alpha', 0.1),
                 name='temporal_weighting'
            ),
            BatchNormalization(name='bn_temp_weight'),
            Dropout(model_cfg.get('dropout_temp_weight', 0.4), name='drop_temp_weight')
        ])

    # --- Recurrent Block ---
    lstm1_units = model_cfg.get('lstm1_units', 128)
    lstm2_units = model_cfg.get('lstm2_units', 64)
    dropout_lstm1 = model_cfg.get('dropout_lstm1', 0.4)
    dropout_lstm2 = model_cfg.get('dropout_lstm2', 0.4)

    lstm_kwargs_1 = {'units': lstm1_units, 'return_sequences': True, 'activation': 'tanh'}
    lstm_kwargs_2 = {'units': lstm2_units, 'return_sequences': True, 'activation': 'tanh'}

    if use_bilstm:
        model_layers.append(Bidirectional(LSTM(**lstm_kwargs_1), name='bilstm1'))
    else:
        model_layers.append(LSTM(**lstm_kwargs_1, name='lstm1'))
    model_layers.append(BatchNormalization(name='bn3'))
    model_layers.append(Dropout(dropout_lstm1, name='drop3'))

    if use_bilstm:
        model_layers.append(Bidirectional(LSTM(**lstm_kwargs_2), name='bilstm2'))
    else:
        model_layers.append(LSTM(**lstm_kwargs_2, name='lstm2'))
    model_layers.append(BatchNormalization(name='bn4'))
    model_layers.append(Dropout(dropout_lstm2, name='drop4'))

    # --- Attention Block ---
    mha_layer = MultiHeadAttention(num_heads=mha_heads, key_dim=mha_key_dim, name='mha1')
    model_layers.append(SelfAttentionWrapper(mha_layer, name='self_attention_wrapper'))

    # --- Pooling Layer ---
    model_layers.append(GlobalAveragePooling1D(name='gap1'))

    # --- Dense Block ---
    model_layers.extend([
        Dense(model_cfg.get('dense1_units', 64), activation='relu', name='dense1'),
        Dropout(model_cfg.get('dropout_dense', 0.4), name='drop5'),
        Dense(num_classes, activation='softmax', name='output')
    ])

    model = Sequential(model_layers)
    return model