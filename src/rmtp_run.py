#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"

import numpy as np
import pandas as pd
import tensorflow as tf
from hawkes_train import _read_data_helper, read_data_wmata
from RMTP_PtProc import RMTP

user_data = _read_data_helper("/home/nik90/datasets/wmata/wmata_2015_2016/user_trips/user_timeseries")

def next_batch(user_data):
    prev_cnt = 0
    stlabels = {}
    for cnt in range(100, len(user_data), 100):
        tm, hl, od, mask, userids, stlabels = read_data_wmata(user_data[prev_cnt : cnt], stationlabels=stlabels)
        prev_cnt = cnt
        seqlen = mask.sum(axis=1)
        data = np.stack([tm, hl, od], axis=-1)
        xin = data[:, :-1, :]
        yout = data[:, 1:, :]
        yout[:, :, 0] = np.diff(data[:, :, 0])
        yield xin, yout, seqlen, stlabels


batchiter = next_batch(user_data)

with tf.Session() as sess:
    rt = RMTP(256,100, 3, input_class_size=[1, 7*24, 3000],
              input_embedding_size=[1, 128, 128],
              clipping_val=6, cell_type=tf.contrib.rnn.BasicLSTMCell)

    rt.init_variables()
    rt.build_graph()
    tf.global_variables_initializer().run(session=sess)
    for batch in batchiter:
        xin, yout, seqlen, stationlabels = batch
        rt_args = {"tf_inputs": {"x_in:0": xin, "y_out:0": yout, "lenmask:0": seqlen},
                   "epochs": 1000,
                   "learning_rate": 1e-6,
                   "reuse": True,
                   "sess": sess}
        rt.run(**rt_args)

    tf.train.Saver().save(sess, "rmtp_all_model.pkl")


