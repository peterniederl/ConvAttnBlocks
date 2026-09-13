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


def _attention_block(channels, attention):
    if attention == "none":
        return None
    if attention == "se":
        return SEBlock(channels, reduction=8)
    if attention == "cbam":
        return CBAM(channels)
    raise ValueError("attention must be one of: none, se, cbam")


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
    x = layers.Conv2D(stem_channels, 3, strides=1, padding="same", use_bias=False)(inputs)
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
