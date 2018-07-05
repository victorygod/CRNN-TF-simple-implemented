import model

if __name__== '__main__':
	crnn = model.CRNN()
	# crnn.train("train/")
	crnn.infer("val/")