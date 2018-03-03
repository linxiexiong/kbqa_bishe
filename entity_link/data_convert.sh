#!/bin/sh

# gen train datas

#nohup python3 -u convert_data.py ../datas/topic_words/small_topic_words_1w.txt ../datas/SimpleQuestions_v2/small_train_1w.txt ../datas/SimpleQuestions_v2/small_train_whole_1w.csv train &
nohup python3 -u convert_data.py ../datas/topic_words/train.fuzzy_p2_linker.simple_linker.original.union ../datas/SimpleQuestions_v2/annotated_fb_data_train.txt ../datas/SimpleQuestions_v2/all_train_whole.csv train &

#gen test datas

# nohup python3 -u convert_data '../datas/topic_words/small_topic_words_test_1w.txt', '../datas/SimpleQuestions_v2/small_test_1w.txt', '../datas/SimpleQuestions_v2/small_test_whole_1w.csv', 'test' &
#nohup python3 -u convert_data.py ../datas/topic_words/test.fuzzy_p2_linker.simple_linker.original.union ../datas/SimpleQuestions_v2/annotated_fb_data_test.txt ../datas/SimpleQuestions_v2/all_test_whole.csv test &
