# League-of-legends-op.gg-scrapper-Build-Prediction-with-Multi-Label-Classification
## Usage
* You can download and open the <b>python file</b> on your preferred editor.
* You can download and open the <b>notebook</b> on Jupyter Notebook or [Google Colab](https://colab.research.google.com/).

## Workflow
1. Inspect the op.gg page
2. go to EUW server and learderboard
3. Choose a parser (lxml , html5lib , html.parser)
4. Create a beautifulsoup object
5. Extract top players urls and store it in a list
6. access each player's match history and get match infos and the items bought
7. Make a dataframe
8. Download a CSV file that contains all data scraped

## Specification
The scrapers are different between one site and another. So, to use those scrapers, you have to change the value of <i>base_site</i> with the url desired, and identify tags to extract.

## Packages used
```python
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import datetime from datetime
import pandas as pd
```

### ML & Data Mining:
# Approach:
Aiming to create a generalized solution by taking into account all the important parameters for a the matches. This will give the user a much more personalized experience and user can get a recommnendation from the best.

# Step 1: Data gathering : -merge all the dataframes obtained from regular scrapping 
# Step 2: Preprocessing : cleaning the dataset,mapping the results and game modes for better processing, only took games with kda_ratio higher than 2.6:1
# Step 3: merged current dataset with an updated items details dataset and another champions details
# Step 3 : 
# Recommendation system with NLP
compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each item explanation
Now to calculate similarity score,we did a bit of research on which are the easiest methods. We found several techniques like euclidean, the Pearson and the cosine similarity scores. 
We will be using cosine similarity scores to find the similarity between two movies
One advantage of cosine similarity score is that it is independent of magnitude and is relatively easy and fast to calculate.

# Step 4: Create a get_recommendation function which takes an item name from the user and recommends similar items used by pro playersby taking the cosine simalarity scores of the most similar item explanation.

## Packages used
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
