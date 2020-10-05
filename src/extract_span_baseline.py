from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os

import tensorflow as tf
import numpy as np
import span_util
import util
import h5py
from custom_coref import CustomCorefIndependent

if __name__ == "__main__":
    config = util.initialize_from_env()
    log_dir = config["log_dir"]

    if sys.argv[1] == "train_bert_base":
        embed_dim = 768
    elif sys.argv[1] == "train_bert_large":
        embed_dim = 1024

    # Input file in .jsonlines format.
    input_filename = sys.argv[2]

    # Span embeddings will be written to this file in .h5 format.
    output_dir = sys.argv[3]
    output_prefix = sys.argv[4]

    model = CustomCorefIndependent(config)
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.restore_init(session)

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
                parent_child_emb_pos = span_util.get_parent_child_emb_baseline(pos_clusters, candidate_span_emb, candidate_starts, candidate_ends, "positive", embed_dim)
                parent_child_emb_neg = span_util.get_parent_child_emb_baseline(neg_clusters, candidate_span_emb, candidate_starts, candidate_ends, "negative", embed_dim)
                if parent_child_emb_pos is None and parent_child_emb_neg is not None:
                    parent_child_list.extend([parent_child_emb_neg])
                elif parent_child_emb_neg is None and parent_child_emb_pos is not None:
                    parent_child_list.extend([parent_child_emb_pos])
                elif parent_child_emb_pos is not None and parent_child_emb_neg is not None:
                    parent_child_list.extend([parent_child_emb_pos, parent_child_emb_neg])

                if (example_num+1) % 350 == 0 or (example_num+1) == num_lines:
                    write_count += 1
                    filename = output_prefix + "_" + str(write_count) + ".h5"
                    out_filename = os.path.join(output_dir, filename)
                    print('Writing files: {}'.format(out_filename))
                    sys.stdout.flush()
                    parent_child_reps = tf.concat(parent_child_list, 0).eval()
                    with h5py.File(out_filename, 'w') as hf:
                        hf.create_dataset("span_representations", data=parent_child_reps, compression="gzip", compression_opts=0, shuffle=True, chunks=True)
                    parent_child_list = []