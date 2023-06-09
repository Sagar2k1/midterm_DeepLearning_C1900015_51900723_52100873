import tensorflow as tf
from position import positional_encoding_1d, positional_encoding_2d, point_wise_feed_forward_network

MAX_LENGTH = 50
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, emb_dim, num_heads, fc_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderBlock, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                      key_dim=emb_dim,
                                                      dropout=dropout_rate)

        self.ffn = point_wise_feed_forward_network(emb_dim=emb_dim,
                                                   fc_dim=fc_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
            `mask` is padding mask
        """
        attn_output = self.mha(query=x, key=x, value=x,
                               training=training, attention_mask=mask)

        # (batch_size, input_seq_len, fully_connected_dim)
        out1 = self.layernorm1(attn_output + x, training=training)

        # (batch_size, input_seq_len, fully_connected_dim)
        ffn_output = self.ffn(out1, training=training)

        ffn_output = self.dropout1(ffn_output, training=training)

        # (batch_size, input_seq_len, fully_connected_dim)
        encoder_layer_out = self.layernorm2(ffn_output + out1, training=training)

        return encoder_layer_out
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 num_layers,
                 emb_dim,  # Input/output dimensionality (or Embedding dim).
                 num_heads,
                 fc_dim,  # Inner-layer dimensionality (or FC dim).
                 row_size, col_size,    # Shape of grid features
                 dropout_rate=0.1,
                 layernorm_eps=1e-6):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_layers = num_layers

        # Embeddings (it's just a Dense layer)
        self.embedding = tf.keras.layers.Dense(emb_dim, activation='relu')
        # Positional encoding 2D
        self.pos_encoding = positional_encoding_2d(row_size, col_size, emb_dim)

        # Encoder layers.
        self.enc_layers = [EncoderBlock(emb_dim=emb_dim,
                                        num_heads=num_heads,
                                        fc_dim=fc_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps)
            for _ in range(num_layers)]

        # Dropout.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        # Sum up embeddings and positional encoding.
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]

        # Add dropout.
        x = self.dropout(x, training=training)

        # N encoder blocks.
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # Shape `(batch_size, input_seq_len, emb_dim)

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 emb_dim,  # Input/output dimensionality (or Embedding dim).
                 num_heads,
                 fc_dim,  # Inner-layer dimensionality (or FC dim).
                 dropout_rate=0.1,
                 layernorm_eps=1e-6):
        super().__init__()

        # Masked multi-head self-attention.
        self.mha_masked = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            # Size of each attention head for query Q and key K.
            key_dim=emb_dim,
            dropout=dropout_rate
        )
        # Multi-head cross-attention.
        self.mha_cross = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            # Size of each attention head for query Q and key K.
            key_dim=emb_dim,
            dropout=dropout_rate
        )

        # Point-wise feed-forward network.
        self.ffn = point_wise_feed_forward_network(emb_dim, fc_dim)

        # Layer normalization.
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        # Dropout for the point-wise feed-forward network.
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # The encoder output shape is `(batch_size, input_seq_len, emb_dim)`.

        attn_masked, attn_weights_masked = self.mha_masked(
            query=x,
            value=x,
            key=x,
            # A boolean mask that prevents attention to certain positions.
            attention_mask=look_ahead_mask,
            # Shape `(batch_size, target_seq_len, emb_dim)`.
            return_attention_scores=True,
            training=training
        )

        out1 = self.layernorm1(attn_masked + x)

        attn_cross, attn_weights_cross = self.mha_cross(
            query=out1,
            value=enc_output,
            key=enc_output,
            # A boolean mask that prevents attention to certain positions.
            attention_mask=padding_mask,
            # Shape `(batch_size, target_seq_len, emb_dim)`.
            return_attention_scores=True,
            training=training
        )

        out2 = self.layernorm2(attn_cross + out1)

        # Shape `(batch_size, target_seq_len, emb_dim)`.
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout1(ffn_output, training=training)
        
        # Shape `(batch_size, target_seq_len, emb_dim)`.
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_masked, attn_weights_cross

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 num_layers,
                 emb_dim,  # Input/output dimensionality.
                 num_heads,
                 fc_dim,  # Inner-layer dimensionality.
                 target_vocab_size,
                 dropout_rate=0.1,
                 layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size,
            emb_dim,
            mask_zero=True
        )
        self.pos_encoding = positional_encoding_1d(MAX_LENGTH, emb_dim)

        self.dec_layers = [
            DecoderBlock(
                emb_dim=emb_dim,
                num_heads=num_heads,
                fc_dim=fc_dim,
                dropout_rate=dropout_rate,
                layernorm_eps=layernorm_eps)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Sum up embeddings and positional encoding.
        # Shape: `(batch_size, target_seq_len, emb_dim)`.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.emb_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # The shape of x is `(batch_size, target_seq_len, emb_dim)`.
        return x, attention_weights

