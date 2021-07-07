import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class movie_recommender:

	def combine_features(self, row):
		combine = ''
		for feature in self.features:
			combine +=row[feature]+" "
		return combine	

	def __init__(self):

		self.df = pd.read_csv('movie_dataset.csv')
		self.features = ['keywords', 'original_language', 'cast', 'director','genres']

		for feature in self.features:
			self.df[feature] = self.df[feature].fillna('')

		self.df['combined'] = self.df.apply(self.combine_features, axis=1)

		cv = CountVectorizer()
		cv_fit = cv.fit_transform(self.df['combined'])

		self.similarity = cosine_similarity(cv_fit.toarray())

	def get_index(self,title):
		try:
			return self.df[self.df.title==title]['index'].values[0]
		except:
			print(f"{title} not found in Database" )
	
	def get_title(self,index):
		return self.df[self.df.index==index]['title'].values[0]


	def get_recommendations(self,Movie):
		movie_index = self.get_index(Movie)
		similar_movie = list(enumerate(self.similarity[movie_index]))
		similar_movie = sorted(similar_movie, key=lambda x:x[1], reverse=True)

		for i in range(0,10):
			print(self.get_title(similar_movie[i][0]))



def main():
	
	Recommender = movie_recommender()
	Movie = ''
	while Movie!='END':
		Movie = input("Movie you like (type 'END' to end program) : ")
		if Movie == 'END':
			print('Exiting Program')
			break
		else:
			Recommender.get_recommendations(Movie)

if __name__ == "__main__":
    main()