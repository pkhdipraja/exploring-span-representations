from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os

import tensorflow as tf
import numpy as np
import span_util_predict
import util
import h5py
from custom_coref import CustomCorefIndependent

# python3 extract_span_baseline_predict.py bert_base test.english.128.probe_reduced.jsonlines test_bert_base_baseline_128
# python3 extract_span_baseline_predict.py bert_large test.english.384.probe_reduced.jsonlines test_bert_base_baseline_384
if __name__ == "__main__":
    config = util.initialize_from_env()
    log_dir = config["log_dir"]

    if sys.argv[1] == "bert_base":
        embed_dim = 768
    elif sys.argv[1] == "bert_large":
        embed_dim = 1024

    # Input file in .jsonlines with extension
    # "test.english.128.probe_reduced.jsonlines"
    input_filename = sys.argv[2]

    # Output filename without extention, because both h5 and jsonlines file will be named that
    # e.g. test.english.128.probe_reduced_output
    output_filename = sys.argv[3]

    # input_filename = '../data/test.english.128.probe_reduced.jsonlines'
    # output_filename = '../data/test.english.128.probe_reduced_output'

    model = CustomCorefIndependent(config)
    saver = tf.train.Saver()

    # write_count = 0
    output_filename_json = output_filename + ".jsonlines"
    output_filename_h5 = output_filename + ".h5"

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.restore_init(session)

        with open(output_filename_json, 'w') as output_file:
            with open(input_filename) as input_file:
                parent_child_list = []
                write_count = 0
                num_lines = sum(1 for line in input_file.readlines())
                input_file.seek(0)  # return to first line
                for example_num, line in enumerate(input_file.readlines()):
                    example = json.loads(line)
                    tensorized_example = model.tensorize_example(example, is_training=False)
                    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
                    candidate_span_emb, candidate_starts, candidate_ends = session.run(model.embeddings, feed_dict=feed_dict)
                    candidate_span_emb = candidate_span_emb[:, :2*embed_dim]  # exclude attention head and span features
                    pos_clusters, neg_clusters = example["distances_positive"], example["distances_negative"]

                    # get_parent_child_emb returns info_dict(to create a json file with unique doc key and sentences)
                    # and parent_child_emb to create an h5 file with
                    # parent_child_emb, mention_dist, men1_start, men1_end, men2_start, men2_end, doc_key, gold_label

                    info_dict_pos, parent_child_emb_pos = span_util_predict.get_parent_child_emb_baseline(pos_clusters, candidate_span_emb, candidate_starts, candidate_ends, "positive", embed_dim)
                    info_dict_neg, parent_child_emb_neg = span_util_predict.get_parent_child_emb_baseline(neg_clusters, candidate_span_emb, candidate_starts, candidate_ends, "negative", embed_dim)
                    if parent_child_emb_neg is not None:
                        # Add the neg sample to dataset
                        parent_child_list.extend([parent_child_emb_neg])
                        # add sentence strings to info json
                        info_dict_neg['sentences'] = example["sentences"]
                        # write line of json with info with doc_key and sentences
                        output_file.write(json.dumps(info_dict_neg))
                        output_file.write("\n")
                    if parent_child_emb_pos is not None:
                        # add only pos examples to dataset
                        parent_child_list.extend([parent_child_emb_pos])
                        info_dict_pos['sentences'] = example["sentences"]
                        output_file.write(json.dumps(info_dict_pos))
                        output_file.write("\n")

                    if example_num % 100 == 0:
                        print("Decoded {} examples.".format(example_num + 1))

                    if (example_num + 1) % 350 == 0 or (example_num + 1) == num_lines:
                        # write_count += 1
                        print('Writing files: {}'.format(output_filename_h5))
                        sys.stdout.flush()
                        parent_child_reps = tf.concat(parent_child_list, 0).eval()
                        with h5py.File(output_filename_h5, 'w') as hf:
                            hf.create_dataset("span_representations", data=parent_child_reps, compression="gzip",
                                              compression_opts=0, shuffle=True, chunks=True)
                        parent_child_list = []
