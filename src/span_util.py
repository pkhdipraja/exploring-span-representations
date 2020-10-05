from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import numpy as np


def get_parent_child_emb(clusters, span_emb, span_starts, span_ends, label_type):
    """
    Make [n, 2*emb + 2] tensor where n is number of samples and emb is dimension of embeddings
    and last 2 values are distance between mentions and gold label, respectively.
    """
    span_emb_list = []
    dist_list = []
    for coref_relation, dist in clusters:
        assert len(coref_relation) == 2, 'Member of mentions are not equal to 2'
        parent_idx = np.intersect1d(np.where(span_starts == coref_relation[0][0]), 
                                    np.where(span_ends == coref_relation[0][1]))
        child_idx = np.intersect1d(np.where(span_starts == coref_relation[1][0]),
                                   np.where(span_ends == coref_relation[1][1]))
        if len(parent_idx) == 1 and len(child_idx) == 1:  # There are some mentions that exceeded max_span_width, this check skips such mentions.
            parent_child_span = tf.concat([span_emb[parent_idx], span_emb[child_idx]], 1)
            span_emb_list.append(parent_child_span)
            dist_list.append(dist)

    if span_emb_list:
        parent_child_emb = tf.concat(span_emb_list, 0)
        mention_dist = tf.dtypes.cast(tf.stack(dist_list, 0), tf.float32)
        mention_dist = tf.reshape(mention_dist, [-1,1])

        if label_type == "positive":
            gold_label = tf.ones([parent_child_emb.shape[0],1], tf.float32)
        elif label_type == "negative":
            gold_label = tf.zeros([parent_child_emb.shape[0],1], tf.float32)
        return tf.concat([parent_child_emb, mention_dist, gold_label], 1)
    else:
        return None


def get_parent_child_emb_baseline(clusters, span_emb, span_starts, span_ends, label_type, embed_dim):
    """
    Make [n, 2*max_span_width*embed_dim + 2] tensor where n is number of samples
    max_span_width is maximum width of span allowed from configuration files,
    embed_dim is dimension of BERT embeddings, and last 2 values are distance between
    mentions and gold label, respectively
    """
    span_emb_list = []
    dist_list = []
    max_span_width = 30

    for coref_relation, dist in clusters:
        assert len(coref_relation) == 2, 'Member of mentions are not equal to 2'
        parent_intersect_idx = np.intersect1d(np.where(span_starts == coref_relation[0][0]), 
                                              np.where(span_ends == coref_relation[0][1]))
        child_intersect_idx = np.intersect1d(np.where(span_starts == coref_relation[1][0]), 
                                             np.where(span_ends == coref_relation[1][1]))
        if parent_intersect_idx:         
            parent_idx = np.intersect1d(np.where(span_starts == coref_relation[0][0]), 
                                        np.where(span_ends <= coref_relation[0][1]))
        else:
            parent_idx = []

        if child_intersect_idx:
            child_idx = np.intersect1d(np.where(span_starts == coref_relation[1][0]), 
                                       np.where(span_ends <= coref_relation[1][1]))
        else:
            child_idx = []
   
        span_dist_1 = coref_relation[0][1] - coref_relation[0][0] + 1
        span_dist_2 = coref_relation[1][1] - coref_relation[1][0] + 1
        span_flag = span_dist_1 <= max_span_width and span_dist_2 <= max_span_width

        if len(parent_idx) > 0 and len(child_idx) > 0 and span_flag:
            # parent_start_emb = tf.reshape(span_emb[parent_idx][0,:embed_dim], [1,-1]) # take embedding of first wordpieces
            # parent_body_emb = tf.reshape(span_emb[parent_idx][:,embed_dim:], [1,-1])  # take embedding of all wordpieces after the first one (until end wordpieces) and flatten
            # parent_emb = tf.concat([parent_start_emb, parent_body_emb], 1)
            parent_emb = tf.reshape(span_emb[parent_idx][:, embed_dim:], [1, -1])

            # child_start_emb = tf.reshape(span_emb[child_idx][0,:embed_dim], [1,-1]) 
            # child_body_emb = tf.reshape(span_emb[child_idx][:,embed_dim:], [1,-1])
            # child_emb = tf.concat([child_start_emb, child_body_emb], 1)
            child_emb = tf.reshape(span_emb[child_idx][:, embed_dim:], [1, -1])

            # Pad token representations w.r.t to max span width
            parent_paddings = [[0, 0], [0, max_span_width*embed_dim - tf.shape(parent_emb)[1]]]
            child_paddings = [[0, 0], [0, max_span_width*embed_dim - tf.shape(child_emb)[1]]]
            parent_emb = tf.pad(parent_emb, parent_paddings, "CONSTANT")
            child_emb = tf.pad(child_emb, child_paddings, "CONSTANT")
            parent_child_span = tf.concat([parent_emb, child_emb], 1)
            span_emb_list.append(parent_child_span)
            dist_list.append(dist)

    if span_emb_list:
        parent_child_emb = tf.concat(span_emb_list, 0)
        mention_dist = tf.dtypes.cast(tf.stack(dist_list, 0), tf.float32)
        mention_dist = tf.reshape(mention_dist, [-1,1])

        if label_type == "positive":
            gold_label = tf.ones([parent_child_emb.shape[0],1], tf.float32)
        elif label_type == "negative":
            gold_label = tf.zeros([parent_child_emb.shape[0],1], tf.float32)
        return tf.concat([parent_child_emb, mention_dist, gold_label], 1)
    else:
        return None