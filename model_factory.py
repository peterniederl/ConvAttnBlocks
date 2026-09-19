import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CBAM(layers.Layer):
    def __init__(self, channels, reduction=8, spatial_kernel=7, **kwargs):
        super().__init__(**kwargs)
        hidden = max(channels // reduction, 1)
        self.channel_mlp = keras.Sequential([
            layers.Dense(hidden, activation="gelu"),
            layers.Dense(channels),
        ])
        self.spatial = layers.Conv2D(1, spatial_kernel, padding="same", activation="sigmoid")

    def call(self, inputs):
        average = tf.reduce_mean(inputs, axis=[1, 2])
        maximum = tf.reduce_max(inputs, axis=[1, 2])
        channel_gate = tf.nn.sigmoid(self.channel_mlp(average) + self.channel_mlp(maximum))
        channel_gate = channel_gate[:, None, None, :]
        x = inputs * channel_gate
        spatial_average = tf.reduce_mean(x, axis=-1, keepdims=True)
        spatial_maximum = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_gate = self.spatial(tf.concat([spatial_average, spatial_maximum], axis=-1))
        return x * spatial_gate

class SEBlock(layers.Layer):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(
            channels // reduction,
            activation="gelu",
            use_bias=True
        )
        self.fc2 = layers.Dense(
            channels,
            activation="sigmoid",
            use_bias=True
        )

    def call(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = tf.reshape(s, [-1, 1, 1, x.shape[-1]])
        return x * s


class _AxialSpatialGate(layers.Layer):
    def __init__(self, fusion, use_pointwise=False, use_post_pointwise=False, **kwargs):
        super().__init__(**kwargs)
        self.fusion = fusion
        self.use_pointwise = use_pointwise
        self.use_post_pointwise = use_post_pointwise

    def build(self, input_shape):
        height, width = input_shape[1:3]
        channels = input_shape[-1]
        if height is None or width is None or channels is None:
            raise ValueError("AxialSpatialGate requires known spatial dimensions")
        self.pointwise = (
            layers.Conv2D(channels, 1, padding="same", use_bias=False)
            if self.use_pointwise else None
        )
        self.vertical = layers.DepthwiseConv2D(
            (height, 1), padding="valid", use_bias=False
        )
        self.horizontal = layers.DepthwiseConv2D(
            (1, width), padding="valid", use_bias=False
        )
        self.vertical_post_pointwise = (
            layers.Conv2D(channels, 1, padding="same", use_bias=False)
            if self.use_post_pointwise else None
        )
        self.horizontal_post_pointwise = (
            layers.Conv2D(channels, 1, padding="same", use_bias=False)
            if self.use_post_pointwise else None
        )
        self.vertical_norm = layers.LayerNormalization(axis=-1)
        self.horizontal_norm = layers.LayerNormalization(axis=-1)
        super().build(input_shape)

    def call(self, inputs):
        projected = inputs if self.pointwise is None else self.pointwise(inputs)
        vertical_features = self.vertical(projected)
        horizontal_features = self.horizontal(projected)
        if self.use_post_pointwise:
            vertical_features = self.vertical_post_pointwise(vertical_features)
            horizontal_features = self.horizontal_post_pointwise(horizontal_features)
        vertical_features = self.vertical_norm(vertical_features)
        horizontal_features = self.horizontal_norm(horizontal_features)
        if self.fusion == "multiply":
            fused = vertical_features * horizontal_features
        else:
            fused = vertical_features + horizontal_features
        gate = tf.nn.sigmoid(fused)
        return inputs * gate


class AxialSpatialGateMultiply(_AxialSpatialGate):
    def __init__(self, **kwargs):
        super().__init__(fusion="multiply", **kwargs)


class AxialSpatialGateMultiplyPointwise(_AxialSpatialGate):
    def __init__(self, **kwargs):
        super().__init__(fusion="multiply", use_pointwise=True, **kwargs)


class AxialSpatialGateSum(_AxialSpatialGate):
    def __init__(self, **kwargs):
        super().__init__(fusion="sum", **kwargs)


class AxialSpatialGateSumPointwise(_AxialSpatialGate):
    def __init__(self, **kwargs):
        super().__init__(fusion="sum", use_pointwise=True, **kwargs)


class AxialSpatialGateMultiplyPostPointwise(_AxialSpatialGate):
    def __init__(self, **kwargs):
        super().__init__(fusion="multiply", use_post_pointwise=True, **kwargs)


class AxialSpatialGateSumPostPointwise(_AxialSpatialGate):
    def __init__(self, **kwargs):
        super().__init__(fusion="sum", use_post_pointwise=True, **kwargs)


class AxialSumSEBlock(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.axial_sum = AxialSpatialGateSum()
        self.se = SEBlock(channels, reduction=8)

    def call(self, inputs):
        return self.se(self.axial_sum(inputs))


class SEAxialSumBlock(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.se = SEBlock(channels, reduction=8)
        self.axial_sum = AxialSpatialGateSum()

    def call(self, inputs):
        return self.axial_sum(self.se(inputs))


class AxialAvgPool2ConvGateSum(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vertical_dw1 = layers.DepthwiseConv2D((7, 1), padding="same", use_bias=False)
        self.vertical_dw2 = layers.DepthwiseConv2D((7, 1), padding="same", use_bias=False)
        self.horizontal_dw1 = layers.DepthwiseConv2D((1, 7), padding="same", use_bias=False)
        self.horizontal_dw2 = layers.DepthwiseConv2D((1, 7), padding="same", use_bias=False)
        self.vertical_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.horizontal_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def call(self, inputs):
        vertical_path = tf.reduce_mean(inputs, axis=2, keepdims=True)
        vertical_path = self.vertical_dw1(vertical_path)
        vertical_path = tf.nn.gelu(vertical_path)
        vertical_path = self.vertical_dw2(vertical_path)
        vertical_path = self.vertical_norm(vertical_path)

        horizontal_path = tf.reduce_mean(inputs, axis=1, keepdims=True)
        horizontal_path = self.horizontal_dw1(horizontal_path)
        horizontal_path = tf.nn.gelu(horizontal_path)
        horizontal_path = self.horizontal_dw2(horizontal_path)
        horizontal_path = self.horizontal_norm(horizontal_path)

        fused = vertical_path + horizontal_path
        gate = tf.nn.sigmoid(fused)
        return inputs * gate


class AxialConvAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        height, width = input_shape[1:3]
        channels = input_shape[-1]
        if height is None or width is None or channels is None:
            raise ValueError("AxialConvAttention requires known spatial dimensions")
        self.channels = channels

        self.vertical_1 = layers.DepthwiseConv2D(
            (height, 1), padding="valid", use_bias=False
        )
        self.horizontal_1 = layers.DepthwiseConv2D(
            (1, width), padding="valid", use_bias=False
        )
        self.vertical_2 = layers.DepthwiseConv2D(
            (height, 1), padding="valid", use_bias=False
        )
        self.horizontal_2 = layers.DepthwiseConv2D(
            (1, width), padding="valid", use_bias=False
        )
        
        self.key_norm = layers.LayerNormalization(axis=-1)
        self.query_norm = layers.LayerNormalization(axis=-1)
        super().build(input_shape)

    def call(self, inputs):
        key_vertical = self.vertical_1(inputs)
        key_horizontal = self.horizontal_1(inputs)
        key = key_vertical + key_horizontal
        key = self.key_norm(key)

        query_vertical = self.vertical_2(inputs)
        query_horizontal = self.horizontal_2(inputs)
        query = query_vertical + query_horizontal
        query = self.query_norm(query)

        attention_logits = query * key
        attention_logits = attention_logits / tf.sqrt(tf.cast(self.channels, inputs.dtype))
        attn = tf.nn.sigmoid(attention_logits)

        return inputs * attn


class AxialConvSelfAttention(layers.Layer):
    def __init__(self, heads=1, use_projections=True, **kwargs):
        super().__init__(**kwargs)
        if heads < 1:
            raise ValueError("heads must be at least 1")
        if not use_projections and heads != 1:
            raise ValueError("The no-projection variant supports one head")
        self.heads = heads
        self.use_projections = use_projections
        self.channels = None
        self.head_dim = None
        self.scale = None

    def build(self, input_shape):
        height, width = input_shape[1:3]
        channels = input_shape[-1]
        if height is None or width is None or channels is None:
            raise ValueError("AxialConvSelfAttention requires known spatial dimensions")
        if channels % self.heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.head_dim = channels // self.heads if self.use_projections else channels
        self.scale = self.head_dim ** -0.5

        self.vertical_1 = layers.DepthwiseConv2D(
            (height, 1), padding="valid", use_bias=False, depth_multiplier=self.heads
        )
        self.horizontal_1 = layers.DepthwiseConv2D(
            (1, width), padding="valid", use_bias=False, depth_multiplier=self.heads
        )
        self.vertical_2 = layers.DepthwiseConv2D(
            (height, 1), padding="valid", use_bias=False, depth_multiplier=self.heads
        )
        self.horizontal_2 = layers.DepthwiseConv2D(
            (1, width), padding="valid", use_bias=False, depth_multiplier=self.heads
        )
        if self.use_projections:
            self.query_projection = layers.Conv2D(
                self.heads * self.head_dim, 1, padding="same", use_bias=False
            )
            self.key_projection = layers.Conv2D(
                self.heads * self.head_dim, 1, padding="same", use_bias=False
            )
            self.value_conv = layers.Conv2D(
                self.heads * self.head_dim, 7, padding="same", use_bias=False
            )
            self.output_projection = layers.Conv2D(
                channels, 1, padding="same", use_bias=False
            )
        else:
            self.query_projection = None
            self.key_projection = None
            self.value_conv = layers.DepthwiseConv2D(
                7, padding="same", use_bias=False
            )
            self.output_projection = None
        
        self.key_norm = layers.LayerNormalization(axis=-1)
        self.query_norm = layers.LayerNormalization(axis=-1)
        super().build(input_shape)

    def call(self, inputs):
        key_vertical = self.vertical_1(inputs)
        key_horizontal = self.horizontal_1(inputs)
        key = key_vertical + key_horizontal
        if self.key_projection is not None:
            key = self.key_projection(key)
        key = self.key_norm(key)

        query_vertical = self.vertical_2(inputs)
        query_horizontal = self.horizontal_2(inputs)
        query = query_vertical + query_horizontal
        if self.query_projection is not None:
            query = self.query_projection(query)
        query = self.query_norm(query)

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        query = tf.reshape(query, [batch_size, height, width, self.heads, self.head_dim])
        key = tf.reshape(key, [batch_size, height, width, self.heads, self.head_dim])
        value = self.value_conv(inputs)
        value = tf.reshape(value, [batch_size, height, width, self.heads, self.head_dim])

        horizontal_scores = tf.einsum("bhwgc,bhvgc->bhwvg", query, key)
        horizontal_scores *= self.scale
        horizontal_attention = tf.nn.softmax(horizontal_scores, axis=3)
        horizontal_output = tf.einsum(
            "bhwvg,bhvgc->bhwgc", horizontal_attention, value
        )

        vertical_scores = tf.einsum("bhwgc,bivgc->bhwig", query, key)
        vertical_scores *= self.scale
        vertical_attention = tf.nn.softmax(vertical_scores, axis=3)
        output = tf.einsum("bhwig,bivgc->bhwgc", vertical_attention, horizontal_output)
        output = tf.reshape(output, [batch_size, height, width, self.heads * self.head_dim])
        if self.output_projection is not None:
            output = self.output_projection(output)

        return output


class AxialConvSelfAttentionNoProjection(AxialConvSelfAttention):
    def __init__(self, **kwargs):
        super().__init__(heads=1, use_projections=False, **kwargs)




class AxialFullConvGate(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vertical = None
        self.horizontal = None
        self.vertical_refine = layers.DepthwiseConv2D(
            (1, 7), padding="same", use_bias=False
        )
        self.horizontal_refine = layers.DepthwiseConv2D(
            (7, 1), padding="same", use_bias=False
        )
        self.vertical_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.horizontal_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def build(self, input_shape):
        height, width = input_shape[1:3]
        if height is None or width is None:
            raise ValueError("AxialFullConvGate requires known spatial dimensions")
        self.vertical = layers.DepthwiseConv2D(
            (height, 1), padding="valid", use_bias=False
        )
        self.horizontal = layers.DepthwiseConv2D(
            (1, width), padding="valid", use_bias=False
        )
        super().build(input_shape)

    def call(self, inputs):
        vertical_path = self.vertical(inputs)
        vertical_path = tf.nn.gelu(vertical_path)
        vertical_path = self.vertical_refine(vertical_path)
        vertical_path = self.vertical_norm(vertical_path)

        horizontal_path = self.horizontal(inputs)
        horizontal_path = tf.nn.gelu(horizontal_path)
        horizontal_path = self.horizontal_refine(horizontal_path)
        horizontal_path = self.horizontal_norm(horizontal_path)

        fused = vertical_path + horizontal_path
        gate = tf.nn.sigmoid(fused)
        return inputs * gate


def _attention_block(channels, attention):
    if attention == "none":
        return None
    if attention == "se":
        return SEBlock(channels, reduction=8)
    if attention == "cbam":
        return CBAM(channels)
    if attention == "axial_multiply":
        return AxialSpatialGateMultiply()
    if attention == "axial_multiply_pointwise":
        return AxialSpatialGateMultiplyPointwise()
    if attention == "axial_sum":
        return AxialSpatialGateSum()
    if attention == "axial_sum_pointwise":
        return AxialSpatialGateSumPointwise()
    if attention == "axial_multiply_postpointwise":
        return AxialSpatialGateMultiplyPostPointwise()
    if attention == "axial_sum_postpointwise":
        return AxialSpatialGateSumPostPointwise()
    if attention == "axial_sum_se":
        return AxialSumSEBlock(channels)
    if attention == "se_axial_sum":
        return SEAxialSumBlock(channels)
    if attention == "axial_conv_attention":
        return AxialConvAttention()
    if attention == "axial_conv_self_attention":
        return AxialConvSelfAttention(heads=4)
    if attention == "axial_conv_self_attention_no_projection":
        return AxialConvSelfAttentionNoProjection()
    if attention == "axial_avg_pool_dual_norm":
        return AxialAvgPool2ConvGateSum()
    if attention == "axial_full_conv_gate":
        return AxialFullConvGate()
    raise ValueError(
        "attention must be one of: none, se, cbam, axial_multiply, "
        "axial_multiply_pointwise, axial_sum, axial_sum_pointwise, "
        "axial_multiply_postpointwise, axial_sum_postpointwise, "
        "axial_sum_se, se_axial_sum, axial_avg_pool_dual_norm, "
        "axial_conv_attention, axial_conv_self_attention, "
        "axial_conv_self_attention_no_projection, "
        "axial_full_conv_gate"
    )


class ResidualBlock(layers.Layer):
    def __init__(self, filters, stride=1, attention="none", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.attention_name = attention
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.attention = None
        self.projection = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if self.stride != 1 or input_channels != self.filters:
            self.projection = keras.Sequential([
                layers.Conv2D(self.filters, 1, strides=self.stride, padding="same", use_bias=False),
                layers.BatchNormalization(),
            ])
        self.attention = _attention_block(self.filters, self.attention_name)
        super().build(input_shape)

    def call(self, inputs, training=None):
        shortcut = inputs if self.projection is None else self.projection(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if self.attention is not None:
            x = self.attention(x)
        return tf.nn.relu(x + shortcut)


def build_resnet(config):
    inputs = keras.Input(shape=(config.image_size, config.image_size, 3))
    model_config = config.model
    stem_channels = model_config["stem_channels"]
    stage_channels = model_config["stage_channels"]
    stage_depths = model_config["stage_depths"]
    x = layers.Conv2D(stem_channels, 5, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for stage_index, (channels, depth) in enumerate(zip(stage_channels, stage_depths)):
        for block_index in range(depth):
            stride = 2 if stage_index > 0 and block_index == 0 else 1
            x = ResidualBlock(channels, stride=stride, attention=config.attention)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(model_config["dropout_rate"])(x)
    outputs = layers.Dense(config.num_classes)(x)
    return keras.Model(inputs, outputs, name=f"resnet_{config.attention}")
