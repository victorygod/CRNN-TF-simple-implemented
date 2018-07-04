import tensorflow as tf
import numpy as np
import os, utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

activation = tf.nn.leaky_relu

class CRNN:
	def __init__(self):
		self.batch_size = 20
		self.num_classes = utils.num_classes
		self.inputs = tf.placeholder(tf.float32, [self.batch_size, 64, 1200, 1])
		self.labels = tf.sparse_placeholder(tf.int32)
		self.dense_decoded = None
		self.loss = None
		self.train_op = None
		self.accuracy = None
		self.build()

	def build(self):
		with tf.variable_scope("CNN"):
			net = tf.layers.conv2d(self.inputs, name = "conv1_1", filters = 64, kernel_size = 5, strides = (1,1), padding = "same", activation = None)
			net = tf.layers.batch_normalization(net, training = True)
			net = activation(net)
			net = tf.layers.conv2d(net, name = "conv1_2", filters = 64, kernel_size = 5, strides = (1,1), padding = "same", activation = None)
			net = tf.layers.batch_normalization(net, training = True)
			net = activation(net)
			net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, padding = "SAME")
			
			net = tf.layers.conv2d(net, name = "conv2_1", filters = 128, kernel_size = 5, strides = (1,1), padding = "same", activation = None)
			net = tf.layers.batch_normalization(net, training = True)
			net = activation(net)
			net = tf.layers.conv2d(net, name = "conv2_2", filters = 128, kernel_size = 5, strides = (1,1), padding = "same", activation = None)
			net = tf.layers.batch_normalization(net, training = True)
			net = activation(net)
			net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, padding = "SAME")

			net = tf.layers.conv2d(net, name = "conv3_1", filters = 256, kernel_size = 5, strides = (1,1), padding = "same", activation = None)
			net = tf.layers.batch_normalization(net, training = True)
			net = activation(net)
			net = tf.layers.conv2d(net, name = "conv3_2", filters = 256, kernel_size = 5, strides = (1,1), padding = "same", activation = None)
			net = tf.layers.batch_normalization(net, training = True)
			net = activation(net)
			net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, padding = "SAME")

			net = tf.layers.conv2d(net, name = "conv4_1", filters = 512, kernel_size = 5, strides = (1,1), padding = "same", activation = None)
			net = tf.layers.batch_normalization(net, training = True)
			net = activation(net)
			net = tf.layers.conv2d(net, name = "conv4_2", filters = 512, kernel_size = 5, strides = (1,1), padding = "same", activation = None)
			net = tf.layers.batch_normalization(net, training = True)
			net = activation(net)
			net = tf.layers.max_pooling2d(net, pool_size = 2, strides = 2, padding = "SAME")

			net = tf.layers.conv2d(net, name = "conv5_1", filters = 512, kernel_size = 4, strides = (1, 1), padding = "valid", activation = activation)

			_, feature_h, feature_w, _ = net.get_shape().as_list()
			
			net = tf.transpose(net, [0, 2, 1, 3])
			net = tf.reshape(net, (self.batch_size, feature_w, feature_h * 512))

			seq_len = tf.fill([net.get_shape().as_list()[0]], feature_w)

		with tf.variable_scope("RNN1"):
			lstm_fw_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(256)
			lstm_bw_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(256)
			net, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, net, seq_len, dtype=tf.float32)
			net = tf.concat(net, 2)
		with tf.variable_scope("RNN2"):
			lstm_fw_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(256)
			lstm_bw_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(256)
			net, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, net, seq_len, dtype=tf.float32)
			net = tf.concat(net, 2)

		with tf.variable_scope("last_layer"):
			net = tf.reshape(net, [-1, 512])
			net = tf.layers.dense(net, self.num_classes)
			net = tf.reshape(net, [self.batch_size, -1, self.num_classes])
			net = tf.transpose(net, (1, 0, 2))

			decoded, log_prob = tf.nn.ctc_beam_search_decoder(net, seq_len)
			self.dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value = -1)

			self.global_step = tf.train.get_or_create_global_step()
			loss = tf.nn.ctc_loss(self.labels, net, seq_len)
			self.loss = tf.reduce_mean(loss)
			self.train_op = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=self.global_step)
			self.accuracy = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.labels))

			tf.summary.scalar('loss', self.loss)
			self.merged_summay = tf.summary.merge_all()
			
	def train(self, src):
		data_loader = utils.DataLoader(src)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(max_to_keep=1)
			train_writer = tf.summary.FileWriter('summary_ckpt/', sess.graph)
			ckpt = tf.train.latest_checkpoint(utils.checkpoint_dir)
			if ckpt:
				saver.restore(sess, ckpt)
				print('restore from checkpoint{0}'.format(ckpt))
			print('=============================begin training=============================')
			train_cost = 0
			last_step = 0
			while data_loader.epoch<100:
				batch_inputs, batch_labels, sparse_labels = data_loader.next_batch(self.batch_size)
				
				feed_dict = {self.inputs: batch_inputs, self.labels: batch_labels}
				summary_str, loss, step, _ = sess.run([self.merged_summay, self.loss, self.global_step, self.train_op], feed_dict)
				# calculate the cost
				train_cost += loss
				train_writer.add_summary(summary_str, step)
				print(step, loss)

				if step % 100 == 1:
					accuracy, decoded = sess.run([self.accuracy, self.dense_decoded], feed_dict)
					print(step, accuracy, train_cost/(step - last_step))
					with open("accuracy.txt", "w") as f:
						f.write(str(step) + ' ' + str(accuracy) + ' ' + str(train_cost/(step - last_step)) + '\n')
						for s in decoded:
							for c in s:
								if c>0:
									f.write(utils.decode_maps[c])
							f.write("\n")
						f.write("===================================\n")
						for s in sparse_labels:
							for c in s:
								if c>0:
									f.write(utils.decode_maps[c])
							f.write("\n")
					last_step = step
					train_cost = 0

				if step % 100 == 0:
					saver.save(sess, os.path.join(utils.checkpoint_dir, 'ocr-model'), global_step=step)

			saver.save(sess, os.path.join(utils.checkpoint_dir, 'ocr-model'), global_step=step)