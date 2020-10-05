import json


def get_error_count(input_filename):
    '''
    :param input_filename: predicted jsonlines filepath
    :return: error_counter, num_examples, num of true_pos, true_neg, false_pos, false_neg

    one line of input file:
    dict_keys(['mention_dist', 'men1_end', 'sentences', 'men1_start', 'pred', 'men2_start', 'men2_end', 'gold_label', 'doc_key'])
    '''
    error_counter = 0
    num_examples = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    # with open(errors, 'w'):
    with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
            example = json.loads(line)
            # for each gold label in cluster check it against pred label
            for idx, label in enumerate(example['gold_label']):
                pred_label = example['pred'][idx]
                # calculate number of all mention pairs
                num_examples += 1

                # only a subset of example for error analysis
                if num_examples < 750:

                    # calculate true positives, false positives...
                    if label == 1 and pred_label == 1:
                        true_pos += 1
                    elif label == 0 and pred_label == 0:
                        true_neg +=1
                    elif label == 1 and pred_label == 0:
                        false_neg +=1
                    elif label == 0 and pred_label == 1:
                        false_pos += 1

                    # output errors
                    if label != pred_label:
                        error_counter += 1
                        sentences = [item for sublist in example['sentences'] for item in sublist]
                        print(example['mention_dist'][idx], 'pred:', example['pred'][idx], 'gold:', example['gold_label'][idx])
                        before = max(0, example['men1_start'][idx] - 10)
                        after = min(len(sentences)-1, example['men2_end'][idx]+3)
                        print(" ".join(sentences[example['men1_start'][idx]:example['men1_end'][idx]+1]), '   ==   ', " ".join(sentences[example['men2_start'][idx]:example['men2_end'][idx]+1]))
                        if example['mention_dist'][idx] < 15:
                            print('...', " ".join(sentences[before:after]), "(", example['mention_dist'][idx], ")", '\n' )
                        else:
                            print('...', " ".join(sentences[before:example['men1_end'][idx]+3]), '...', " ".join(sentences[example['men2_start'][idx]-10:after]), "(", example['mention_dist'][idx], ")", '\n' )



    return error_counter, num_examples, true_pos, true_neg, false_pos, false_neg


if __name__ == '__main__':
    filenames = ["pred_test.english.128.probe_reduce.joshi.jsonlines", "pred_test.english.384.probe_reduce.joshi.jsonlines"]

    for input_filename in filenames:
        print(input_filename)
        error_count, num_examples, true_pos, true_neg, false_pos, false_neg  = get_error_count(input_filename)
        print(f'true_pos, true_neg, false_pos, false_neg, {true_pos, true_neg, false_pos, false_neg}')