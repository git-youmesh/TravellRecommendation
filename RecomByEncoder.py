import pandas as pd
from urllib import request
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from gensim.models import Word2Vec
mytoken =" "
login(token=mytoken)
# Get the playlist dataset file
data = request.urlopen('http://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')
# Parse the playlist dataset file. Skip the first two lines as
# they only contain metadata
lines = data.read().decode("utf-8").split('\n')[2:]
# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split())> 1]
# Load song metadata
 
songs_file =request.urlopen('http://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns = ['id', 'title','artist'])
model = Word2Vec(
playlists, vector_size=32, window=20, negative=50,
min_count=1, workers=4
)

song_id = 2172
# Ask the model for songs similar to song #2172
simlarsongs = model.wv.most_similar(positive=str(song_id))
print(simlarsongs)
