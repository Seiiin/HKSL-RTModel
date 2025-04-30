import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class AdaptedUnfoldTemporalWindows1D(layers.Layer):
    """
    Extracts sliding temporal windows from a (B, T, C) tensor using tf.image.extract_patches.
    """
    def __init__(self, window_size=9, window_stride=1, window_dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

    def call(self, x):
        b, t, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x4 = tf.reshape(x, [b, 1, t, c]) # Reshape for extract_patches (needs rank 4)
        patches = tf.image.extract_patches(
            x4,
            sizes=[1, 1, self.window_size, 1],    # Window size in (B, H, W, C) dims
            strides=[1, 1, self.window_stride, 1], # Stride in (B, H, W, C) dims
            rates=[1, 1, self.window_dilation, 1], # Dilation in (B, H, W, C) dims
            padding='SAME')
        # Output shape: (B, 1, T_out, win_size * C)
        t_out = tf.shape(patches)[2]
        patches = tf.reshape(patches, [b, t_out, self.window_size, c]) # Reshape to (B, T_out, W, C)
        return patches

    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        if t is not None and self.window_stride > 0:
            # Calculate output time dimension based on 'SAME' padding and stride
            t_out = tf.cast(tf.math.ceil(tf.cast(t, tf.float32) / tf.cast(self.window_stride, tf.float32)), tf.int32)
        else:
            t_out = None # Keep time dimension dynamic
        return (b, t_out, self.window_size, c)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            window_size=self.window_size,
            window_stride=self.window_stride,
            window_dilation=self.window_dilation)
        return cfg

@tf.keras.utils.register_keras_serializable()
class AdaptedTemporalWeighting1D(layers.Layer):
    """
    Lightweight temporal weighting module inspired by CorrNet+.
    Uses parallel dilated convolutions to derive dynamic temporal weights.
    """
    def __init__(self, reduction_factor=16, num_convs=3,
                 initial_alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.reduction_factor = reduction_factor
        self.num_convs = num_convs
        self.initial_alpha = initial_alpha
        # Layers will be instantiated in build()
        self.dense_transform = None
        self.dense_back = None
        self.conv_enhance = []
        self.enhance_weights = None
        self.alpha = None

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None: raise ValueError("Input channels must be known.")
        hidden = max(1, channels // self.reduction_factor)

        self.dense_transform = layers.Dense(hidden, use_bias=True, name='dense_transform')
        self.dense_back = layers.Dense(channels, use_bias=True, name='dense_back')

        self.conv_enhance = []
        for i in range(self.num_convs):
            # Depthwise-like conv per feature group
            conv = layers.Conv1D(hidden, kernel_size=3, padding='same',
                                 dilation_rate=i + 1, groups=hidden,
                                 name=f'conv_enhance_{i}')
            self.conv_enhance.append(conv)

        # Build sub-layers (needed for TF graph mode)
        dummy_time = input_shape[1] if input_shape[1] is not None else 1
        self.dense_transform.build((None, dummy_time, channels))
        self.dense_back.build((None, dummy_time, hidden))
        for conv in self.conv_enhance:
            conv.build((None, dummy_time, hidden))

        self.enhance_weights = self.add_weight(name='enhance_weights', shape=(self.num_convs,),
                                               initializer=tf.keras.initializers.Constant(1.0 / self.num_convs),
                                               trainable=True)
        self.alpha = self.add_weight(name='alpha', shape=(1,),
                                     initializer=tf.keras.initializers.Constant(self.initial_alpha),
                                     trainable=True)
        super().build(input_shape)

    def call(self, x):
        temporal_summary = self.dense_transform(x) # (B, T, H)

        enhance_outputs = []
        weights = tf.unstack(self.enhance_weights)
        for i in range(self.num_convs):
             enhanced = self.conv_enhance[i](temporal_summary) # (B, T, H)
             enhance_outputs.append(enhanced * weights[i])

        aggregated_out = tf.add_n(enhance_outputs) # (B, T, H)
        out_weights = self.dense_back(aggregated_out) # (B, T, C)

        temporal_gate = tf.nn.sigmoid(out_weights) - 0.5 # Center gate around 0
        weighted_update = x * temporal_gate * self.alpha

        return x + weighted_update

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(reduction_factor=self.reduction_factor, num_convs=self.num_convs,
                   initial_alpha=self.initial_alpha)
        return cfg

@tf.keras.utils.register_keras_serializable()
class AdaptedCorrelation1D(layers.Layer):
    """
    Temporal correlation module inspired by CorrNet+.
    Compares center frame features with neighboring frame features using attention or simpler methods.
    """
    def __init__(self, neighbors=3, reduction_factor=16,
                 initial_alpha=0.0, use_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.neighbors = neighbors
        self.reduction_factor = reduction_factor
        self.initial_alpha = initial_alpha
        self.use_attention = use_attention
        self.window_size = 2 * neighbors + 1

        self.unfold = AdaptedUnfoldTemporalWindows1D(self.window_size)
        # Layers will be instantiated in build()
        self.query_proj = None
        self.key_proj = None
        self.value_proj = None
        self.neighbor_processor = None # Used if use_attention=False
        self.alpha = None

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None: raise ValueError("Input channels must be known.")

        self.unfold.build(input_shape) # Build unfold layer first

        if self.use_attention:
            self.query_proj = layers.Dense(channels, name='query_proj', use_bias=True)
            self.key_proj = layers.Dense(channels, name='key_proj', use_bias=True)
            self.value_proj = layers.Dense(channels, name='value_proj', use_bias=True)
            # Build projection layers
            dummy_input_shape = (None, channels)
            self.query_proj.build(dummy_input_shape)
            self.key_proj.build(dummy_input_shape)
            self.value_proj.build(dummy_input_shape)
        else:
            # Simplified version: Process neighbors before correlation
            hidden_proc = max(1, channels // self.reduction_factor)
            self.neighbor_processor = layers.Dense(hidden_proc, activation='relu',
                                                   name='neighbor_processor', use_bias=True)
            # Build the processor layer (acts on each neighbor feature vector)
            self.neighbor_processor.build((None, channels))

        self.alpha = self.add_weight(name='alpha', shape=(1,),
                                     initializer=tf.keras.initializers.Constant(self.initial_alpha),
                                     trainable=True)
        super().build(input_shape)

    def call(self, x):
        b, t, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        unfolded = self.unfold(x)                         # (B, T, W, C)
        center_idx = self.neighbors
        center = unfolded[:, :, center_idx, :]            # (B, T, C) Center frame features

        # Extract neighbors (excluding center)
        neighbors_before = unfolded[:, :, :center_idx, :] # (B, T, neighbors, C)
        neighbors_after = unfolded[:, :, center_idx + 1:, :] # (B, T, neighbors, C)
        neigh = tf.concat([neighbors_before, neighbors_after], axis=2) # (B, T, W-1, C)
        num_neigh = self.window_size - 1

        if self.use_attention:
            # --- Attention-based Correlation ---
            query = self.query_proj(center)                   # (B, T, C)
            query_expanded = tf.expand_dims(query, axis=2)    # (B, T, 1, C)

            neigh_flat = tf.reshape(neigh, [-1, c])
            keys_flat = self.key_proj(neigh_flat)      # (B*T*num_neigh, C)
            values_flat = self.value_proj(neigh_flat)  # (B*T*num_neigh, C)

            keys = tf.reshape(keys_flat, [b, t, num_neigh, c]) # (B, T, num_neigh, C)
            values = tf.reshape(values_flat, [b, t, num_neigh, c]) # (B, T, num_neigh, C)

            # Batched dot product for attention scores
            attn_scores = tf.einsum('btic,btnc->btin', query_expanded, keys) # (B, T, 1, num_neigh)

            scale = tf.math.sqrt(tf.cast(c, tf.float32))
            safe_scale = tf.where(tf.equal(scale, 0.0), 1.0, scale) # Avoid division by zero
            attn_scores = attn_scores / safe_scale

            # Sigmoid gate centered around 0
            attn_weights = tf.nn.sigmoid(attn_scores) - 0.5 # (B, T, 1, num_neigh)

            # Apply weights to values
            correlation_features = tf.einsum('btin,btnc->btic', attn_weights, values) # (B, T, 1, C)
            correlation_features = tf.squeeze(correlation_features, axis=2) # (B, T, C)

        else:
             # --- Simplified Correlation (Non-Attention) ---
             neigh_flat = tf.reshape(neigh, [-1, c])
             processed_neighbors_flat = self.neighbor_processor(neigh_flat) # (B*T*num_neigh, H_proc)

             h_proc = tf.shape(processed_neighbors_flat)[-1]
             processed_neighbors = tf.reshape(processed_neighbors_flat, [b, t, num_neigh, h_proc]) # (B, T, num_neigh, H_proc)

             center_expanded = tf.expand_dims(center, axis=2) # (B, T, 1, C)

             # Ensure dimensions match for comparison (assuming H_proc == C here)
             if h_proc != c:
                 raise NotImplementedError("Non-attention correlation needs neighbor_processor output dim == C or center projection.")

             # Scaled dot product between center and processed neighbors
             correlation_scores = tf.einsum('btic,btnc->btin', center_expanded, processed_neighbors) # (B, T, 1, num_neigh)

             scale = tf.math.sqrt(tf.cast(c, tf.float32))
             safe_scale = tf.where(tf.equal(scale, 0.0), 1.0, scale)
             correlation_scores = correlation_scores / safe_scale

             # Sigmoid gate centered around 0
             correlation_weights = tf.nn.sigmoid(correlation_scores) - 0.5 # (B, T, 1, num_neigh)

             # Apply weights to the *original* neighbors
             correlation_features = tf.einsum('btin,btnc->btic', correlation_weights, neigh) # (B, T, 1, C)
             correlation_features = tf.squeeze(correlation_features, axis=2) # (B, T, C)


        # Additive update with learned alpha scaling
        output = x + correlation_features * self.alpha
        return output

    def compute_output_shape(self, input_shape):
         return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(neighbors=self.neighbors, reduction_factor=self.reduction_factor,
                   initial_alpha=self.initial_alpha, use_attention=self.use_attention)
        return cfg