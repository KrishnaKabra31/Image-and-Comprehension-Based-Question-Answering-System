from model import *
from keras.utils.np_utils import to_categorical
from nltk.tokenize import word_tokenize
from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

"""
Training protype
""""
ldata = {"cat_1.jpg":{"questions":["what is the color of the cat?", "which animal is on grass?","is the cat on grass?"], "answers" : ["white", "cat", "yes"]},"cat_2.jpg":{"questions":["what is the color of the cat?", "which animal is on the floor?","is the cat on grass?", "is the cat having food?"], "answers" : ["black", "cat", "no", "yes"]},"dog_1.jpg":{"questions":["what is the color of the dog?", "which animal is on grass?","is the dog on grass?", "what is the dog bitting?"], "answers" : ["white", "dog", "yes", "shoe"]},"dog_2.jpg":{"questions":["what is the color of the dog?", "which animal is on floor?","is the dog on grass?", "how many animals are on the floor?"], "answers" : ["brown", "dog", "no","2"]}}

def read_data():
	print("Reading Data")
	imgs = []
	ques = []
	ans_f = []
	data1 = json.load(open("data_prepro.json"))
	idx2word = data1['ix_to_word']
	print(idx2word)
	for i in ldata.keys():
	#	print("img: {}".format(i))
		img = np.array(image.load_img('Train/'+str(i), target_size=(224,224)))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		img = preprocess_input(x)
		#print(data[i])
		for j in ldata[i]['questions']:
			q = [x.lower() for x in word_tokenize(j)]
			que = [0 for x in range(0,26)]
			que_len = len(q)
			print(j)
			loc = 0
			for k in q:
				for l in idx2word.keys():
					if idx2word[l] == k:
						que[loc] = int(l)
						loc+=1
						break
			print(que)	
			que = np.array(que)
			que_len = np.array(que_len)
			que_check = move_right(que, que_len)
			que_check = np.reshape(que_check,(1,26))
			ques.append(que_check)
			imgs.append(img)
		ar = ldata[i]['answers']
		data = json.load(open("data_prepro.json"))
		idx2ans = data['ix_to_ans']
		#print('\n\n\n\n',idx2ans,'\n\n\n\n')
		for j in ar:
			print(j)
		#	print("answers: {}".format(j))
			train_y = [0 for i in range(0,1000)]
			ans = [x.lower() for x in word_tokenize(j)]
			cnt = 0
			print("ans: {}".format(ans))
			for k in idx2ans.keys():
				if idx2ans[k] == ans[0]:
					save = cnt
					break
				cnt+=1
			print("save: {}".format(save))
			train_y[save] = 1
			train_y = np.array(train_y).reshape((1, 1000))
			ans_f.append(train_y)
#	print(ans_f)
#	ans = to_categorical(ans)
	#print("Img shape: {}".format(img.shape))
	imgs = np.array(imgs)
	imgs = imgs.reshape(imgs.shape[0],imgs.shape[2],imgs.shape[3],imgs.shape[4])
	ques = np.array(ques)
	ques = ques.reshape(ques.shape[0], ques.shape[2])
	train_X = [imgs, ques]
	train_y = np.array(ans_f)
	train_y = train_y.reshape(train_y.shape[0], train_y.shape[2])
	#print("Y shape: {}, Y: {}".format(train_y.shape, train_y))
	return train_X, train_y

def train():
	train_X, train_y = read_data()
	meta_data = json.load(open('data_prepro.json', 'r'))
	meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
	num_words = len(meta_data['ix_to_word'])
	num_classes = len(meta_data['ix_to_ans'])
	vqa = model(num_words, 300, num_classes)
	file_name = 'weights-{epoch:02d}.hdf5'
	checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
	vqa.fit(train_X, train_y, epochs = 40, callbacks = [checkpoint], verbose = 1)
	return vqa

if __name__ == "__main__":
	train()
	#train_X, train_y = read_data()
	#print(train_X[1].shape, train_X[0].shape)
	#print(train_y.shape)
#	cust_data()
	#train_X, train_y = read_data()
