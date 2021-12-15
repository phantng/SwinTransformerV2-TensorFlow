import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons import layers as tfa_layers
from tensorflow.keras.initializers import TruncatedNormal, Zeros

from modelling.backbone.utils import _n_tuple


"""
The following parts are different in SwinTransformerV2 compared to V1:
1. A post-norm structure is used instead of pre-norm
2. Scaled cosine attention is used instead of dot product attention:
    Sim(qi, kj) = cos(qi, kj )/t + Bij
where t is a learnable scalar, non-shared across heads and layers. Note,
tau is set larger than 0.01. (tf.Variable)
3. Instead of looking up a parametrized meshgrid (ie. a table of coordinates)
pairwise relative position is put through a log transformation and multiplied
element-wise with sign(coordinates). This is meant to reduce the magnitude
of spatial ratios when fine-tuning on a larger window size. This helps
transfers relative positions more effectively when differing window sizes
are used. 
"""


dense_weight_init = dict(
    kernel_initializer=TruncatedNormal(stddev=.02),
    bias_initializer=Zeros()
)


class MLP(layers.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=tfa_layers.GELU,
                 dropout=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features, **dense_weight_init)
        self.act = act_layer()
        self.fc2 = layers.Dense(out_features, **dense_weight_init)
        self.dropout = layers.Dropout(rate=dropout)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def window_partition(x, window_size):
    """

    :param x: (B, H, W, C)
    :param window_size: (int) window_size
    :return: windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.get_shape().as_list()  # have to use this since .shape returns a tensor
    x = tf.reshape(x, shape=(B,
                             H // window_size, window_size,
                             W // window_size, window_size,
                             C))
    # swap columns to get (-1, window_size, window_size, C)
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, window_size, window_size, C))
    return x


def window_reverse(windows, window_size, H, W):
    """

    :param windows: (num_windows*B, window_size, window_size, C)
    :param window_size: (int) Window size
    :param H: int
    :param W: int
    :return: x: (B, H, W, C)
    """
    B = int(windows.get_shape().as_list()[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, shape=(B,
                                   H // window_size, W // window_size,
                                   window_size, window_size,
                                   -1))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(B, H, W, -1))
    return x


class WindowAttention(layers.Layer):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 bias_meta_dim=128):
        """
        :param dim: int number of input channels
        :param window_size: tuple[int] height and width of window
        :param num_heads: int number of attention heads
        :param qkv_bias: bool add learnable bias to query key value
        :param qk_scale: float override default scale in attention
        :param attn_dropout: float dropout ratio of attention weight
        :param proj_dropout: float dropout ratio of output
        :param bias_meta_dim: int number of hidden units in meta network
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # if qk_scale is not provided the default is head_dim ** -0.5
        self.scale = qk_scale or head_dim ** -0.5

        # not sure if build() is the right way, since input weight is known by the time the layer
        # is instantiated, and this layer is not meant to be used in isolation anyways

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, wH, wW
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, wH*wW, wH*wW
        relative_coords = relative_coords.transpose((1, 2, 0))  # Wh*Ww, Wh*Ww, 2

        # SwinV2 log-spaced coordinates
        relative_coords = tf.cast(relative_coords, dtype=tf.float32)
        # element-wise multiplication
        relative_position_index = tf.math.multiply(
            tf.math.sign(relative_coords),
            tf.math.log(1 + tf.abs(relative_coords))
        )  # relative_position_index.shape == Wh*Ww, Wh*Ww, 2
        self.relative_position_index = tf.Variable(
            initial_value=relative_position_index,
            trainable=False,
            name="relative_position_index"
        )
        self.meta_dense = MLP(in_features=self.num_heads,
                              hidden_features=bias_meta_dim,
                              out_features=self.num_heads,
                              dropout=0.0)

        # define weight sublayers here, since they already have a build method,
        # so they will be built when the outer layer gets built
        dense_weight_init_copy = dense_weight_init.copy()
        if not qkv_bias:
            bias_init = None
        else:
            bias_init = Zeros()
        kernel_init = TruncatedNormal(stddev=.02)
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, kernel_initializer=kernel_init,
                                bias_initializer=bias_init)
        # tau >= 0.01 for scaling the attention mechanism; scalar shape is inferred
        self.tau = tf.math.maximum(tf.Variable(initial_value=0.01,
                                               trainable=True,
                                               dtype=tf.float32),
                                   tf.constant(0.01),
                                   name="tau")

        self.attn_dropout = layers.Dropout(rate=attn_dropout)  # use layer instead of tf.nn.dropout which does not have training/inference toggle
        self.proj = layers.Dense(dim, **dense_weight_init)
        self.proj_dropout = layers.Dropout(rate=proj_dropout)
        self.softmax = layers.Softmax()  # check if this works properly

    def call(self,
             x,
             mask=None):
        """

        :param x: input features with shape (num_win*B, N, C)
        :param mask: (0/-inf) mask with shape (num_win, Wh*Ww, Wh*Ww) or None
        :return:
        """
        B_, N, C = x.get_shape().as_list()
        qkv = self.qkv(x)
        # the 3 is here to separate the color channels from the added depth
        qkv = tf.reshape(qkv,
                         shape=(B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # not sure if tensorflow allows traditional unpacking

        # scaled dot product attention replaced by scaled cosine attention
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        dot_qk = q @ k
        # use tf.linalg and math operations to avoid type issues
        q_l2 = tf.linalg.norm(q, ord="euclidean", name="q_l2_norm")
        k_l2 = tf.linalg.norm(k, ord="euclidean", name="k_l2_norm")
        attn = tf.math.divide(dot_qk, tf.math.multiply(q_l2, k_l2))
        attn = tf.math.divide(attn, self.tau)

        # compute relative position bias and sum to obtain scaled cosine attention
        relative_position_bias = self.meta_dense(self.relative_position_index)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))  # nH, Wh*Ww, Wh*Ww
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            # mask (nW, N, N)
            nW = mask.shape[0]  # number of windows in input
            attn = tf.reshape(
                attn,
                shape=(B_ // nW, nW, self.num_heads, N, N)
            )
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
            attn = attn + tf.cast(mask, dtype=tf.float32)
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)

        x = tf.transpose((attn @ v), perm=(0, 2, 1, 3))
        x = tf.reshape(x, shape=(-1, N, C))
        assert x.shape[0] == B_, f"batch shape {B_} doesnt match"
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

    def extra_repr(self):
        # simple func for displaying layer information
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class DropPath(layers.Layer):
    """
    Taken from Rishigami & nku-shengzheliu's Swin Transformer

    DropPath works differently from Dropout in the sense that it doesn't set individual tensor
    elements to zero, but instead sets the WHOLE BATCH to zero.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs, drop_prob, is_training):
        if (not is_training) or (drop_prob == 0.):
            return inputs

        # compute drop_connect tensor
        random_tensor = 1. - self.drop_prob
        shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
        random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)  # this broadcasts to the tensor
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(inputs, 1. - self.drop_prob) * binary_tensor
        return output

    def call(self, x, training=None):
        return self.drop_path(x, self.drop_prob, training)


class SwinTransformerBlock(layers.Layer):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attn_dropout=0.,
                 bias_meta_dim=128,
                 drop_path=0.,
                 act_layer=tfa_layers.GELU,
                 norm_layer=layers.LayerNormalization):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # if window size is larger than input resolution, we dont partition into windows
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, f"shift size {shift_size} is out of range. win_size = {window_size}"

        # define the attention block used; note: the block definition is unchanged per layer;
        # instead the window inputs are shifted, which comes in later
        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = WindowAttention(
            dim, _n_tuple(2)(self.window_size), num_heads=self.num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_dropout=attn_dropout,
            proj_dropout=dropout, bias_meta_dim=bias_meta_dim
        )

        # define sublayers
        self.drop_path = DropPath(drop_path if drop_path > 0. else 0.)  # DropPath returns identity if drop_prob is 0
        self.norm2 = norm_layer(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # mlp is already weight initialized
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            dropout=dropout
        )

    def build(self, input_shape):
        # tf EagerTensor object does not support item assignment

        # calculate attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros(shape=(1, H, W, 1))  # (1, H, W, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = count
                    count += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = tf.reshape(mask_windows, shape=(-1, self.window_size * self.window_size))
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            # torch code uses mask_fill, which operates on self.tensor and a mask tensor.
            # works similarly to np.where and tf.where. replace non-zero mask values with
            # large negative float, and zero mask values with 0.0
            attn_mask = tf.where(attn_mask != 0, -100., attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask,
                                         trainable=False,
                                         name="attention_mask")
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        """
        By order of application:
        input
        layer norm,
        w-msa/sw-msa,
        additive residual connection from input to msa output
        set merge output as shortcut
        layernorm,
        mlp,
        additive residual connection from shortcut to mlp output
        """
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, f"input feature has the wrong size, L: {L}, H*W: {H*W}"

        shortcut = x

        # previously norm was applied before attention and dense layers
        # note that since norm doesnt change input dims, we don't need to change
        # anything other than the ordering

        # reshape input to batch, h, w, chann
        x = tf.reshape(x, shape=(B, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x,
                shift=(-self.shift_size, -self.shift_size),
                axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, win_size, win_size, C
        x_windows = tf.reshape(x, shape=(-1, self.window_size * self.window_size, C))  # nW*B, win_size*win_size, C

        # W-MSA or SW-MSA depending on whether self shift_size > 0 or not
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(attn_windows, shape=(-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B, H', W', C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x,
                shift=(self.shift_size, self.shift_size),
                axis=(1, 2)
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(B, H * W, C))  # same as B, L, C from input

        # apply layer normalization post attention block
        x = self.norm1(x)

        # FFN, ie. vanilla multilayer perceptron with dropout and layer normalization
        x = shortcut + self.drop_path(x)
        # x becomes the short cut here, then additively merged after applying layernorm -> mlp
        # in V2 norm2 is applied after mlp
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        # simple func for displaying extra information about layer
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(layers.Layer):
    """
    Purpose of this layer is to produce a hierarchical representation of the image.
    Works similarly to typical conv.nets in that the feature map resolutions it
    outputs have successively reduced output dimensions. ( think max pool )
    """
    def __init__(self,
                 input_resolution,
                 dim,
                 norm_layer=layers.LayerNormalization):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False, kernel_initializer=TruncatedNormal)  # torch code also has input_dim as dim * 4 but we dont need to specify that
        self.norm = norm_layer(epsilon=1e-5)

    def call(self, x):
        # x: (B, H*W, C)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has the wrong size, L: {L}, H*W: {H*W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W} are not even"

        x = tf.reshape(x, shape=(B, H, W, C))

        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=(B, -1, 4 * C))

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(layers.Layer):
    """
    A SwinTransformer layer for ONE stage; keep in mind that consecutive stages
    employ shifting the windows. Consists of patch merging layer and a swin block
    """
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attn_dropout=0.,
                 drop_path=0.,
                 norm_layer=layers.LayerNormalization,
                 downsample=None, use_checkpoint=False):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        # assert depth >= 1, "Depth is less than one, which means that there are no Swin Blocks in layer"
        # implement use_checkpoint? or load full model straight?

        # build blocks; check if using list or Sequential is better, if use list: connect every
        # layer manually, if sequential, can just use one call
        # shifting windows in consecutive blocks is done by checking index in range
        # [0, depth) where the first layer always have shift zero, and it is not
        # possible to have depth < 1
        self.blocks = tf.keras.Sequential([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                # implements the shifting windows below for consecutive blocks
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            # first layer has only linear embedding, ie Dense
            self.downsample = None

    def call(self, x):
        """
        Note: ordering here is a bit different from the diagram; the diagram is organized such
        that the downsample layer if present is applied first then the block comes after
        """
        x = self.blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(layers.Layer):
    """
    First layer of SwinTransformer model. Performs patch embedding on image
    """
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 # the C hyperparameter for size of Swin model
                 embed_dim=96,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = _n_tuple(2)(img_size)
        patch_size = _n_tuple(2)(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # note this is not weight initialized with TruncatedNormal
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name="norm_layer")
        else:
            self.norm = None

    def call(self, x):
        # torch code has format CHANNELS_FIRST
        B, H, W, C = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(x, shape=(B, -1, self.embed_dim))
        # torch includes a transpose layer to get channel last format but we dont need that
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2Model(tf.keras.Model):
    """
    TensorFlow 2 implementation of Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/pdf/2103.14030
    """
    def __init__(self,
                 model_name="swin_tiny_patch4_window7_224",
                 include_top=True,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=layers.LayerNormalization,
                 ape=False,
                 patch_norm=True,
                 **kwargs):
        super(SwinTransformerV2Model, self).__init__(name=model_name, **kwargs)
        self.include_top = include_top
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_channels=in_channels, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight(
                name="absolute_position_embedding",
                shape=(1, num_patches, embed_dim),
                initializer=TruncatedNormal(stddev=.02)
            )

        self.pos_drop = layers.Dropout(rate=drop_rate)

        # stochastic depth decay
        # reasoning for using numpy here is that we only need the a list of floats, and
        # TF2 documentation confirms that tf.linspace are identical to np's except when num=0
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]

        # build layers; avoid overwriting .layers attribute
        self.basic_layers = tf.keras.Sequential([
            BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                       input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                         patches_resolution[1] // (2 ** i_layer)),
                       depth=depths[i_layer],
                       num_heads=num_heads[i_layer],
                       window_size=window_size,
                       mlp_ratio=self.mlp_ratio,
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       dropout=drop_rate,
                       attn_dropout=attn_drop_rate,
                       drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                       norm_layer=norm_layer,
                       downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            for i_layer in range(self.num_layers)
        ])

        self.norm = norm_layer(epsilon=1e-5)
        self.avgpool = tfa_layers.AdaptiveAveragePooling1D(1)
        if include_top:
            self.head = layers.Dense(num_classes, **dense_weight_init)
        else:
            self.head = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.basic_layers(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(tf.transpose(x, perm=(0, 2, 1)))  # B C 1
        x = tf.reshape(x, shape=(x.shape[0], -1))
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.include_top:
            x = self.head(x)
        return x


####################################################
# dummy = tf.random.uniform(shape=(16, 224, 224, 3))
# result = SwinTransformerV2Model()(dummy)
# print(result, result.shape)