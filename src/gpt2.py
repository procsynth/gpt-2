#!/usr/bin/env python3

import json
import os

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import numpy as np

import model, sample, encoder
from encoder import Encoder

import tensorflow as tf

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


from threading import Thread
import queue 


class GPT2():
    """GTP-2 helper"""
    def __init__(
        self,
        model_name='1558M',
        seed=None,
        length=None,
        models_dir='models'):

        self.model_name=model_name
        self.seed=seed
        self.length=length
        self.batch_size=1

        self.models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        self.hparams = model.default_hparams()

        with open(os.path.join(self.models_dir, self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = self.hparams.n_ctx // 2
        elif self.length > self.hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.hparams.n_ctx)

        self.tasks = queue.Queue()
        self.model_thread = Thread(target=self.run)
        self.model_thread.start()

    def stop(self):
        self.running = False
        self.generate('dummmy', print)
        print("TF stopping...")


    def run(self):
        
        tf.reset_default_graph()

        enc = self.get_encoder()

        self.running = True

        with tf.Session(graph=tf.Graph()) as sess:

            context = tf.placeholder(tf.int32, [self.batch_size, None])
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

            output = sample.sample_sequence(
                hparams=self.hparams, length=self.length,
                context=context,
                batch_size=self.batch_size,
                temperature=1, top_k=0, top_p=1
            )

            saver = tf.train.Saver()

            print("TF restoring")
            ckpt = tf.train.latest_checkpoint(os.path.join(self.models_dir, self.model_name))
            saver.restore(sess, ckpt)

            while self.running:

                try:
                    task = self.tasks.get(True)
                except Exception as e:
                    self.running = False
                    break

                if not self.running:
                    break

                print("TF processing task", task["run_id"])

                output = sample.sample_sequence(
                    hparams=self.hparams, length=self.length,
                    context=context,
                    batch_size=self.batch_size,
                    temperature=task['temperature'], top_k=task['top_k'], top_p=task['top_p']
                )

                context_tokens = enc.encode(task['raw_text'])

                print("TF len context:", len(context_tokens))

                print("PROMPT:")
                print(task['raw_text'])

                print('=' * 10)

                out = sess.run(output, feed_dict={
                    context: [context_tokens]
                })[:, len(context_tokens):]

                text = enc.decode(out[0])

                task['callback'](text, task)

            print("TF closing")

        print("TF stopped")


    def get_encoder(self):
        with open(os.path.join(self.models_dir, self.model_name, 'encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(os.path.join(self.models_dir, self.model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        return Encoder(
            encoder=encoder,
            bpe_merges=bpe_merges,
        )


    def generate(self, raw_text, callback, run_id=None, nsamples=1, temperature=1, top_k=0, top_p=1):
        self.tasks.put({
            'raw_text': raw_text, 
            'run_id':run_id,
            'callback':callback,
            'nsamples': nsamples,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            })

