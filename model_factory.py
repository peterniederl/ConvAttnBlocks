import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GroupConv2D(layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size=(3,3), padding='same', groups=1, strides=1, kernel_initializer="glorot_uniform", use_bias=True, **kwargs):
        super().__init__(**kwargs)
        assert input_channels % groups == 0, "in_ch must be divisible by groups"
        assert output_channels % groups == 0, "out_ch must be divisible by groups"
        self.in_ch = input_channels
        self.out_ch = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.convs = []
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias

    def build(self, input_shape):
        for _ in range(self.groups):
            self.convs.append(
                layers.Conv2D(self.out_ch // self.groups, self.kernel_size, padding=self.padding, strides=self.strides, kernel_initializer=self.kernel_initializer, use_bias=self.use_bias)
            )

    def call(self, x):
        splits = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        outs = [conv(s) for conv, s in zip(self.convs, splits)]
        return tf.concat(outs, axis=-1)

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


class AxialSumTanh(_AxialSpatialGate):
    def __init__(self, **kwargs):
        super().__init__(fusion="sum", **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer=keras.initializers.Constant(0.1),
            trainable=True,
        )

    def call(self, inputs):
        projected = inputs if self.pointwise is None else self.pointwise(inputs)
        vertical_features = self.vertical(projected)
        horizontal_features = self.horizontal(projected)
        if self.use_post_pointwise:
            vertical_features = self.vertical_post_pointwise(vertical_features)
            horizontal_features = self.horizontal_post_pointwise(horizontal_features)
        vertical_features = self.vertical_norm(vertical_features)
        horizontal_features = self.horizontal_norm(horizontal_features)
        fused = vertical_features + horizontal_features
        gate = 1.0 + self.gamma * tf.tanh(fused)
        return inputs * gate


class AxialFused(_AxialSpatialGate):
    def __init__(self, **kwargs):
        super().__init__(fusion="sum", **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=keras.initializers.Constant(0.5),
            trainable=True,
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer=keras.initializers.Constant(0.1),
            trainable=True,
        )

    def call(self, inputs):
        projected = inputs if self.pointwise is None else self.pointwise(inputs)
        vertical_features = self.vertical(projected)
        horizontal_features = self.horizontal(projected)
        if self.use_post_pointwise:
            vertical_features = self.vertical_post_pointwise(vertical_features)
            horizontal_features = self.horizontal_post_pointwise(horizontal_features)
        vertical_features = self.vertical_norm(vertical_features)
        horizontal_features = self.horizontal_norm(horizontal_features)
        fused = self.alpha * vertical_features + (1 - self.alpha) * horizontal_features
        gate = 1.0 + self.gamma * tf.tanh(fused)
        return inputs * gate


AxialSpatialGateSumResidual = AxialSumTanh
AxialSpatialGateSumWeighted = AxialFused


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
    def __init__(
        self, use_projection=False, projection_ratio=4, query_fusion="sum", **kwargs
    ):
        super().__init__(**kwargs)
        if projection_ratio < 1:
            raise ValueError("projection_ratio must be at least 1")
        if query_fusion not in {"sum", "multiply"}:
            raise ValueError("query_fusion must be 'sum' or 'multiply'")
        self.use_projection = use_projection
        self.projection_ratio = projection_ratio
        self.query_fusion = query_fusion

    def build(self, input_shape):
        height, width = input_shape[1:3]
        channels = input_shape[-1]
        if height is None or width is None or channels is None:
            raise ValueError("AxialConvAttention requires known spatial dimensions")
        self.channels = channels
        self.attention_channels = (
            max(channels // self.projection_ratio, 16)
            if self.use_projection else channels
        )
        self.input_projection = (
            layers.Conv2D(
                self.attention_channels, 1, padding="same", use_bias=False
            )
            if self.use_projection else None
        )
        self.gate_projection = (
            layers.Conv2D(channels, 1, padding="same", use_bias=False)
            if self.use_projection else None
        )

        self.vertical_1 = layers.DepthwiseConv2D(
            (height, 1), padding="valid", use_bias=False,
            depth_multiplier=1
        )
        self.horizontal_1 = layers.DepthwiseConv2D(
            (1, width), padding="valid", use_bias=False,
            depth_multiplier=1
        )
        self.vertical_2 = layers.DepthwiseConv2D(
            (height, 1), padding="valid", use_bias=False,
            depth_multiplier=1
        )
        self.horizontal_2 = layers.DepthwiseConv2D(
            (1, width), padding="valid", use_bias=False,
            depth_multiplier=1
        )
        
        self.key_norm = layers.LayerNormalization(axis=-1)
        self.query_norm = layers.LayerNormalization(axis=-1)
        super().build(input_shape)

    def call(self, inputs):
        projected = inputs if self.input_projection is None else self.input_projection(inputs)
        key_vertical = self.vertical_1(projected)
        key_horizontal = self.horizontal_1(projected)
        key = key_vertical + key_horizontal
        key = self.key_norm(key)

        query_vertical = self.vertical_2(projected)
        query_horizontal = self.horizontal_2(projected)
        if self.query_fusion == "multiply":
            query = query_vertical * query_horizontal
        else:
            query = query_vertical + query_horizontal
        query = self.query_norm(query)

        attention_logits = query * key
        attention_logits = attention_logits / tf.sqrt(
            tf.cast(self.attention_channels, inputs.dtype)
        )
        if self.gate_projection is not None:
            attention_logits = self.gate_projection(attention_logits)
        attn = tf.nn.sigmoid(attention_logits)

        return inputs * attn



class DepthwiseGateUpsample(layers.Layer):
    def __init__(
        self, stride=2, **kwargs
    ):
        self.stride = stride
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.squeeze = layers.DepthwiseConv2D(
            (7, 7),
            padding="same",
            use_bias=True,
            strides=(self.stride, self.stride),
            activation="gelu",
        )
        self.dw = layers.DepthwiseConv2D(
            (7, 7), padding="same", use_bias=True, strides=(1, 1)
        )
        self.sigmoid = layers.Activation("sigmoid")
        self.upsample = layers.UpSampling2D(
            size=(self.stride, self.stride), interpolation="nearest"
        )

        super().build(input_shape)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.dw(x)
        x = self.sigmoid(x)
        x = self.upsample(x)
        return inputs * x


class DepthwiseGateUpsampleStride2(DepthwiseGateUpsample):
    def __init__(self, **kwargs):
        super().__init__(stride=2, **kwargs)


class DepthwiseGateUpsampleStride4(DepthwiseGateUpsample):
    def __init__(self, **kwargs):
        super().__init__(stride=4, **kwargs)


class ProjectedGateUpsample(layers.Layer):
    def __init__(
        self, stride=2, reduction_factor=4, **kwargs
    ):
        self.stride = stride
        self.reduction_factor = reduction_factor
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_factor, 16)
        group_count = self.stride * 2
        self.squeeze = GroupConv2D(
            channels,
            reduced_channels,
            kernel_size=(self.stride, self.stride),
            padding="same",
            groups=group_count,
            strides=self.stride,
            use_bias=True,
        )
        self.gelu = layers.Activation("gelu")
        self.dw = layers.DepthwiseConv2D(
            (7, 7), padding="same", use_bias=True, strides=(1, 1), activation="gelu"
        )
        self.pw = layers.Conv2D(channels, 1, padding="same", use_bias=True)
        self.sigmoid = layers.Activation("sigmoid")
        self.upsample = layers.UpSampling2D(
            size=(self.stride, self.stride), interpolation="nearest"
        )

        super().build(input_shape)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.gelu(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.sigmoid(x)
        x = self.upsample(x)
        return inputs * x


class ProjectedGateUpsampleStride2(ProjectedGateUpsample):
    def __init__(self, **kwargs):
        super().__init__(stride=2, **kwargs)


class ProjectedGateUpsampleStride4(ProjectedGateUpsample):
    def __init__(self, **kwargs):
        super().__init__(stride=4, **kwargs)


class GlobalProjectedGateUpsample(layers.Layer):
    def __init__(self, stride=2, reduction_factor=4, **kwargs):
        self.stride = stride
        self.reduction_factor = reduction_factor
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_factor, 16)
        group_count = self.stride * 2

        self.squeeze = GroupConv2D(
            channels,
            reduced_channels,
            kernel_size=(self.stride, self.stride),
            padding="same",
            groups=group_count,
            strides=self.stride,
            use_bias=True,
        )
        self.local_norm = layers.LayerNormalization(axis=-1)
        self.local_dw = layers.DepthwiseConv2D(
            (7, 7), padding="same", use_bias=True, strides=(1, 1), activation="gelu"
        )
        self.local_pw = layers.Conv2D(channels, 1, padding="same", use_bias=True)

        self.global_avg = layers.GlobalAveragePooling2D()
        self.global_max = layers.GlobalMaxPooling2D()
        self.global_mlp = keras.Sequential([
            layers.Dense(reduced_channels, activation="gelu", use_bias=True),
            layers.Dense(reduced_channels, activation="gelu", use_bias=True),
        ])
        self.global_proj = layers.Conv2D(channels, 1, padding="same", use_bias=True)

        self.sigmoid = layers.Activation("sigmoid")
        self.upsample = layers.UpSampling2D(
            size=(self.stride, self.stride), interpolation="nearest"
        )
        super().build(input_shape)

    def call(self, inputs):
        local = self.squeeze(inputs)
        local = self.local_norm(local)
        local = self.local_dw(local)
        local = self.local_pw(local)

        avg = self.global_avg(inputs)
        mx = self.global_max(inputs)
        global_context = tf.concat([avg, mx], axis=-1)
        global_context = self.global_mlp(global_context)
        global_context = tf.reshape(global_context, [-1, 1, 1, global_context.shape[-1]])
        global_context = self.global_proj(global_context)

        gate = local + global_context
        gate = self.sigmoid(gate)
        gate = self.upsample(gate)
        return inputs * gate



class MultiStageSpatialMixer(layers.Layer):
    def __init__(self, stages=2, reduction_factor=4, **kwargs):
        self.stages = stages
        self.reduction_factor = reduction_factor
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        height = input_shape[1]
        reduced_channels = max(channels // self.reduction_factor, 16)
        self.norm = layers.LayerNormalization(axis=-1)
        self.squeeze = layers.Conv2D(
            reduced_channels,
            kernel_size=1,
            padding="same",
            use_bias=True,
        )
        self.mixers = []
        for stage in range(self.stages):
            stride = 2**(stage + 1)
            kernel = stride + 1
            sec_kernel = min(height//stride + 1, 7)
            self.mixers.append(
                keras.Sequential([
                    layers.DepthwiseConv2D(
                        (kernel, kernel), padding="same", use_bias=True, strides=(stride, stride), activation="gelu"
                    ),
                    layers.DepthwiseConv2D(
                        (sec_kernel, sec_kernel), padding="same", use_bias=True, strides=(1, 1)
                    ),
                    layers.UpSampling2D(
                        size=(stride, stride), interpolation="nearest"
                    )
                ])
            )
        self.excite = layers.Conv2D(channels, 1, padding="same", use_bias=True)
        self.gelu = layers.Activation("gelu")
        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = self.norm(inputs)
        x = self.squeeze(x)
        y = [mixer(x) for mixer in self.mixers]
        local = tf.concat(y, axis=-1)
        local = self.excite(local)
        local = self.gelu(local)
        return local * self.gamma + inputs


def _post_block(channels, attention):
    if attention == "multi_stage_spatial_mixer":
        return MultiStageSpatialMixer(stages=2, reduction_factor=4)
    return None


class GlobalProjectedGateUpsampleStride2(GlobalProjectedGateUpsample):
    def __init__(self, **kwargs):
        super().__init__(stride=2, **kwargs)


class GlobalProjectedGateUpsampleStride4(GlobalProjectedGateUpsample):
    def __init__(self, **kwargs):
        super().__init__(stride=4, **kwargs)


class AxialConvAttentionProjected(AxialConvAttention):
    def __init__(self, **kwargs):
        super().__init__(use_projection=True, projection_ratio=4, **kwargs)


class AxialConvAttentionQueryMultiply(AxialConvAttention):
    def __init__(self, **kwargs):
        super().__init__(query_fusion="multiply", **kwargs)



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
    if attention == "axial_sum_residual_gate":
        return AxialSumTanh()
    if attention == "axial_sum_weighted":
        return AxialFused()
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
    if attention == "axial_conv_attention_projected":
        return AxialConvAttentionProjected()
    if attention == "axial_conv_attention_query_multiply":
        return AxialConvAttentionQueryMultiply()
    if attention == "axial_conv_self_attention":
        return AxialConvAttentionProjected()
    if attention == "axial_conv_self_attention_no_projection":
        return AxialConvAttention()
    if attention == "depthwise_gate_upsample_stride2":
        return DepthwiseGateUpsampleStride2()
    if attention == "depthwise_gate_upsample_stride4":
        return DepthwiseGateUpsampleStride4()
    if attention == "projected_gate_upsample_stride2":
        return ProjectedGateUpsampleStride2()
    if attention == "projected_gate_upsample_stride4":
        return ProjectedGateUpsampleStride4()
    if attention == "global_projected_gate_upsample_stride2":
        return GlobalProjectedGateUpsampleStride2()
    if attention == "global_projected_gate_upsample_stride4":
        return GlobalProjectedGateUpsampleStride4()
    if attention == "axial_avg_pool_dual_norm":
        return AxialAvgPool2ConvGateSum()
    if attention == "axial_full_conv_gate":
        return AxialFullConvGate()
    if attention == "multi_stage_spatial_mixer":
        return None
    raise ValueError(
        "attention must be one of: none, se, cbam, axial_multiply, "
        "axial_multiply_pointwise, axial_sum, axial_sum_residual_gate, "
        "axial_sum_weighted, axial_sum_pointwise, "
        "axial_multiply_postpointwise, axial_sum_postpointwise, "
        "axial_sum_se, se_axial_sum, depthwise_gate_upsample_stride2, "
        "depthwise_gate_upsample_stride4, projected_gate_upsample_stride2, "
        "projected_gate_upsample_stride4, global_projected_gate_upsample_stride2, "
        "global_projected_gate_upsample_stride4, axial_avg_pool_dual_norm, "
        "axial_conv_attention, axial_conv_attention_projected, "
        "axial_conv_attention_query_multiply, "
        "axial_conv_self_attention, "
        "axial_conv_self_attention_no_projection, multi_stage_spatial_mixer, "
        "axial_full_conv_gate"
    )


class ResidualBlock(layers.Layer):
    def __init__(
        self, filters, stride=1, attention="none", use_post_block=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.attention_name = attention
        self.use_post_block = use_post_block
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.attention = None
        self.post_block = None
        self.projection = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if self.stride != 1 or input_channels != self.filters:
            self.projection = keras.Sequential([
                layers.Conv2D(self.filters, 1, strides=self.stride, padding="same", use_bias=False),
                layers.BatchNormalization(),
            ])
        self.attention = _attention_block(self.filters, self.attention_name)
        if self.use_post_block:
            self.post_block = _post_block(self.filters, self.attention_name)
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
        x = tf.nn.relu(x + shortcut)
        if self.post_block is not None:
            x = self.post_block(x)
        return x


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
            is_second_to_last = block_index == depth - 2
            x = ResidualBlock(
                channels,
                stride=stride,
                attention=config.attention,
                use_post_block=is_second_to_last,
            )(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(model_config["dropout_rate"])(x)
    outputs = layers.Dense(config.num_classes)(x)
    return keras.Model(inputs, outputs, name=f"resnet_{config.attention}")
