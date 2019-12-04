# Vertex class
class Vertex:
	def __init__(self, index):
		self.index = index
		self.has_name = False
		self.has_class = False
		self.has_feature = False

	def set_name(self, name):
		self.name = name
		self.has_name = True

	def set_class(self, class_):
		self.class_ = class_
		self.has_class = True

	def set_vocab(self, vocab):
		assert len(vocab) > 0
		self.vocab = vocab
		self.has_feature = True

	def set_word_value(self, word_value):
		assert len(word_value) == len(self.vocab)
		assert self.has_feature == True
		self.word_value = word_value