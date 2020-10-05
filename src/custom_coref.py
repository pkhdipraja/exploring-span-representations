from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os

import tensorflow as tf
import util

from independent import CorefModel
from bert import modeling


class CustomCorefIndependent(CorefModel):
    """
    Modification of Coref model in independent.py to extract span embeddings
    for all possible span (with specified max span width) in the documents.
    """
    def __init__(self, config):
        super(CustomCorefIndependent, self).__init__(config)
        self.embeddings = self.get_idx_span(*self.input_tensors)

    def get_idx_span(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map):
        model = modeling.BertModel(
          config=self.bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          use_one_hot_embeddings=False,
          scope='bert')
        all_encoder_layers = model.get_all_encoder_layers()
        mention_doc = model.get_sequence_output()

        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)

        num_sentences = tf.shape(mention_doc)[0]
        max_sentence_length = tf.shape(mention_doc)[1]
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)
        num_words = util.shape(mention_doc, 0)
        antecedent_doc = mention_doc

        flattened_sentence_indices = sentence_map
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]
        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]), flattened_candidate_mask) # [num_candidates]

        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids) # [num_candidates]

        candidate_span_emb = self.get_span_emb(mention_doc, mention_doc, candidate_starts, candidate_ends) # [num_candidates, emb]

        return [candidate_span_emb, candidate_starts, candidate_ends]

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts # [k]

        if self.config["use_features"]:
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                span_width_index = span_width - 1 # [k]
                span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]], initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index) # [k, emb]
                span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
                span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
                head_attn_reps = tf.matmul(mention_word_scores, context_outputs) # [K, T]
                span_emb_list.append(head_attn_reps)

        span_emb = tf.concat(span_emb_list, 1) # [k, emb]
        return span_emb  # [k, emb]

    def restore_init(self, session):
        # Use train_bert_x as experiment name
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, self.config['tf_checkpoint'])
        init_from_checkpoint = tf.train.init_from_checkpoint
        print("Restoring from {}".format(self.config['tf_checkpoint']))
        init_from_checkpoint(self.config['init_checkpoint'], assignment_map)