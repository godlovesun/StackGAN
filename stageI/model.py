from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.custom_ops import leaky_rectify
from misc.config import cfg


class CondGAN(object):
    def __init__(self, tvs_shape, track_structure_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.tvs_shape = tvs_shape
        self.track_structure_shape = track_structure_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim_tvs = cfg.GAN.DF_DIM_TVS
        self.df_dim_track = cfg.GAN.DF_DIM_TRACK
        self.ef_dim_spec = cfg.GAN.EMBEDDING_DIM_SPEC
        self.ef_dim_pitch = cfg.GAN.EMBEDDING_DIM_PITCH

        self.s_h = self.tvs_shape[0]
        self.s_w = self.tvs_shape[1]
        self.s2_h, self.s4_h, self.s8_h, self.s16_h =\
            int(self.s_h / 2), int(self.s_h / 4), int(self.s_h / 8), int(self.s_h / 16)
        self.s2_w, self.s4_w, self.s8_w, self.s16_w =\
            int(self.s_w / 2), int(self.s_w / 4), int(self.s_w / 8), int(self.s_w / 16)

        # Since D is only used during training, we build a template
        # for safe reuse the variables during computing loss for fake/real/wrong images
        # We do not do this for G,
        # because batch_norm needs different options for training and testing
        if cfg.GAN.NETWORK_TYPE == "default":
            with tf.variable_scope("d_net"):
                self.d_encode_x_template = self.d_encode_x()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        elif cfg.GAN.NETWORK_TYPE == "simple":
            with tf.variable_scope("d_net"):
                self.d_encode_x_template = self.d_encode_x_simple()
                self.d_context_template = self.context_embedding()
                self.discriminator_template = self.discriminator()
        else:
            raise NotImplementedError

    # g-net
    # c_var: [bs, 100, 100, 1]
    def generate_spec_condition(self, c_var_spec):
        embeddings = \
            (pt.wrap(c_var_spec).
             custom_conv2d(self.gf_dim, k_h=5, k_w=5, d_h=3, d_w=3).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_fully_connected(self.ef_dim_spec * 2). # reshape inside the customized method
             apply(leaky_rectify, leakiness=0.2))
        mean = embeddings[:, :self.ef_dim_spec]
        log_sigma = embeddings[:, self.ef_dim_spec:]
        return mean, log_sigma

    def generate_pitch_condition(self, c_var_pitch):
        embeddings = \
            (pt.wrap(c_var_pitch).
             custom_fully_connected(self.ef_dim_pitch * 2).
             apply(leaky_rectify, leakiness=0.2))
        mean = embeddings[:, :self.ef_dim_pitch]
        log_sigma = embeddings[:, self.ef_dim_pitch:]
        return mean, log_sigma

    def generator(self, z_var):
        node1_0 =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16_h * self.s16_w * self.gf_dim * 8).
             fc_batch_norm().
             reshape([-1, self.s16_h, self.s16_w, self.gf_dim * 8]))
        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1). # stride=1 and default padding = SAME, shape conserved
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             # custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s8_h, self.s8_w]).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        backbone_tensor = \
            (node2.
             # custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s4_h, self.s4_w]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s2_h, self.s2_w]).
             custom_conv2d(self.tvs_shape[-1], k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu))

        tvs_output_tensor = \
            (pt.wrap(backbone_tensor).
             # custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s_h, self.s_w]).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))

        track_structure_tensor = \
            (pt.wrap(backbone_tensor).
             flatten().
             custom_fully_connected(self.track_structure_shape[-1]).
             fc_batch_norm().
             apply(tf.nn.relu))
             
        return tvs_output_tensor, track_structure_tensor

    def generator_simple(self, z_var):
        backbone_tensor =\
            (pt.wrap(z_var).
             flatten().
             custom_fully_connected(self.s16_h * self.s16_w * self.gf_dim * 8).
             reshape([-1, self.s16_h, self.s16_w, self.gf_dim * 8]).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s8_h, self.s8_w, self.gf_dim * 4], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             # custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s4_h, self.s4_w, self.gf_dim * 2], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             # custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s2_h, self.s2_w, self.gf_dim], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             # custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu))

        tvs_output_tensor = \
            (pt.wrap(backbone_tensor).
             custom_deconv2d([0] + list(self.tvs_shape), k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             # custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))

        track_structure_tensor = \
            (pt.wrap(backbone_tensor).
             flatten().
             custom_fully_connected(self.track_structure_shape[-1]).
             fc_batch_norm().
             apply(tf.nn.relu))
        return tvs_output_tensor, track_structure_tensor

    # Output of generator will match the shape of self.tvs_shape and self.track_structure_shape
    def get_generator(self, z_var):
        if cfg.GAN.NETWORK_TYPE == "default":
            return self.generator(z_var)
        elif cfg.GAN.NETWORK_TYPE == "simple":
            return self.generator_simple(z_var)
        else:
            raise NotImplementedError

    # d-net
    def context_embedding(self):
        template_spec = (pt.template("input_spec_embed").
                         custom_fully_connected(self.ef_dim_d).
                         apply(leaky_rectify, leakiness=0.2))
        template_pitch = (pt.template("input_pitch").
                          custom_fully_connected(self.ef_dim_pitch).
                          apply(leaky_rectify, leakiness=0.2))
        template = template_spec.concat(1, template_pitch)

        return template

    def d_encode_x(self):
        node1_0 = \
            (pt.template("input_tvs").
             custom_conv2d(self.df_dim_tvs, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim_tvs * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim_tvs * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim_tvs * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim_tvs * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim_tvs * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim_tvs * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        node2 = \
            (pt.template('input_track').
             custom_fully_connected(self.df_dim_track * self.s16_h, self.s16_w).
             fc_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             reshape([-1, self.s16_h, self.s16_w, self.df_dim_track]))
        
        node_out = node1.concat(1, node2)

        return node_out

    def d_encode_x_simple(self):
        node1 = \
            (pt.template("input_tvs").
             custom_conv2d(self.df_dim_tvs, k_h=4, k_w=4). # default stride=2, every conv2d half the shape
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim_tvs * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim_tvs * 4, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim_tvs * 8, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2))

        node2 = \
            (pt.template('input_track').
             custom_fully_connected(self.df_dim_track * self.s16_h, self.s16_w).
             fc_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             reshape([-1, self.s16_h, self.s16_w, self.df_dim_track]))

        node_out = node1.concat(3, node2)

        return node_out

    def discriminator(self):
        template = \
            (pt.template("input").
             custom_conv2d(self.df_dim_tvs * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # same shape as input
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16_h, k_w=self.s16_w, d_h=self.s16_h, d_w=self.s16_w))

        return template

    def get_discriminator(self, x_var_tvs, x_var_track_structure, c_var_spec_embed, c_var_pitch_embed):
        x_code = self.d_encode_x_template.construct(input_tvs = x_var_tvs, input_track = x_var_track_structure)

        c_code = self.d_context_template.construct(input_spec_embed = c_var_spec_embed, input_pitch = c_var_pitch_embed)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16_h, self.s16_w, 1])

        x_c_code = tf.concat(3, [x_code, c_code])
        return self.discriminator_template.construct(input=x_c_code)
