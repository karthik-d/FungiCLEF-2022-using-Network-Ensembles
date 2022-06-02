import keras
from classification_models.keras import Classifiers

ResNeXt101, preprocess_input = Classifiers.get('resnext101')

class InputSequencer(tf.keras.utils.Sequence):

	def __init__(self, base_path=None, csv_name=None, shuffle=True):
		self.BATCH_SIZE = BATCH_SIZE
		self.IMG_SIZE = IMG_SIZE
		self.shuffle = shuffle
		if csv_name is None:
			self.csv_filename = "DF20-train_metadata.csv"
		else:
			self.csv_filename = csv_name
		self.x_col_name = "image_path"
		self.y_col_name = "class_id"
		self.check = []
		print(os.getcwd())
		self.data_file = pd.read_csv(self.csv_filename)
		self.data_file.head()
		print("Classes:", max(self.data_file.class_id.unique())+1)
		self.num_data_pts = len(self.data_file)
		print(self.num_data_pts)
		self.base_path = base_path
		#self.indexes = np.arange(len(self.image_paths))
		self.on_epoch_end()

	def on_epoch_end(self, *args):
		self.check = []
		pass
		"""
		if(self.shuffle):
			np.random.shuffle(self.indexes)
		"""

	def __len__(self):
		return self.num_data_pts // self.BATCH_SIZE
		pass

	def __getitem__(self, idx):
		"""Returns tuple (input, target) correspond to batch #idx."""
		#
		# data_rows = self.data_file.sample(n=self.BATCH_SIZE,replace=False)
		# batch_paths = data_rows[self.x_col_name].to_list()
		# batch_labels = data_rows[self.y_col_name].to_list()
		# batch_labels = list(data_rows.loc[:, [self.y_col_name]])
		if self.base_path is None:
			base_path="/usr/home/bharathi/snake_clef2022"
		else:
			base_path = self.base_path
		'''
		# Karthik
		base_path = os.path.join('/', 'content', 'drive', 'MyDrive', 'Research', 'LifeCLEF\'22', 'SnakeCLEF-2022', 'Dataset', 'SNAKE_CLEF', 'SnakeCLEF2022-small_size', 'SnakeCLEF2022-small_size')
		'''

		batch_images = []
		batch_labels = []
		# The resize error may be occuring because the file is not found and `img` holds None
		# Adding file existence check
		while len(batch_images)<self.BATCH_SIZE:
			
			new_row = self.data_file.sample(n=1, replace=False)
	 		
			path = new_row[self.x_col_name].to_list()[0]
			label = new_row[self.y_col_name].to_list()[0]
			
			if(path in self.check):
				#print("check")
				continue
			else:
				#print("append")
				self.check.append(path)
			
			#print(os.path.isfile(os.path.join(base_path, path)))
			#img=io.imread(os.path.join(base_path, path))
			try:
				img = Image.open(os.path.join(base_path, path)).convert('RGB')
			except FileNotFoundError:
				continue
				
			img_res = preprocess_input(img)
			#img_res = img.resize(self.IMG_SIZE)			
			
			# print(os.path.join(base_path,path))
			image_data = np.array(np.asarray(img_res), dtype='uint8')
			batch_images.append(image_data)
			batch_labels.append(label)
			# print(f"{len(batch_images)} of {self.BATCH_SIZE} images prepared")
			print(path)
	 	
		return (np.array(batch_images), np.array(batch_labels))

data_path = os.path.join('Datasets/DF20')
data_reader_train = InputSequencer(base_path=data_path)
data_reader_test = InputSequencer(base_path='Datasets/DF21-images-300/DF21_300', csv_name="FungiCLEF2022_test_metadata.csv")

opt = keras.optimizers.Adam()
n_classes = 1604
# build model
base_model = ResNeXt101(input_shape=(224,224,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

# train
model.compile(optimizer=opt, data_reader_test, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data_reader_train, validation_data=data_reader_test)
