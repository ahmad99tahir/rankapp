# main.py

from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import spacy
from spacy.matcher import Matcher

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Define stopwords
stop_words = set(stopwords.words('english'))

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Improved clean_text function with stemming and negation handling
def clean_text(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = re.sub(r'\\[rn]', ' ', resumeText)  # remove \r and \n
    return resumeText.strip()  # strip leading and trailing whitespace

def extract_experience(text):
    try:
        experience_score = 0
        matcher = Matcher(nlp.vocab)
        pattern = [{"IS_DIGIT": True, "OP": "?"}, {"LOWER": {"IN": ["to", "or"]}, "OP": "?"}, {"IS_DIGIT": True, "OP": "?"}, {"LOWER": {"IN": ["year", "yr", "yrs", "year's","years"]}}]
        matcher.add("EXPERIENCE", [pattern])
        doc = nlp(text)
        for match_id, start, end in matcher(doc):
            start_token = doc[start]
            if start_token.like_num:
                experience_score = int(start_token.text)
                break
            if start_token.text.isdigit() and doc[end - 1].text.isdigit():
                start_year = int(start_token.text)
                end_year = int(doc[end - 1].text)
                experience_score = end_year - start_year
                break
        return experience_score
    except Exception as e:
        print("Error occurred:", e)
        return None

def extract_education(text):
    education_score = 0
    doc = nlp(text)
    degree_patterns = {
        "bachelor": re.compile(r"\b(b\.?a\.?|b\.?s\.?c?\.?|b\.?e\.?|b\.?tech|bachelor's?)\b"),
        "master": re.compile(r"\b(m\.?a\.?|m\.?s\.?c?\.?|m\.?e\.?|m\.?phil|masters?)\b"),
        "phd": re.compile(r"\b(ph\.?d\.?|doctorate|d\.?phil)\b")
    }
    
    for ent in doc.ents:
        if ent.label_ == "DEGREE" or ent.text.lower() in degree_patterns.keys():
            for degree, pattern in degree_patterns.items():
                if re.search(pattern, ent.text.lower()):
                    if degree == "bachelor":
                        education_score += 1
                    elif degree == "master":
                        education_score += 2
                    elif degree == "phd":
                        education_score += 3
                    break
        elif ent.label_ == "ORG" and ("university" in ent.text.lower() or "college" in ent.text.lower()):
            education_score += 1
    return education_score

def calculate_keyword_score(text, keywords, weights):
    score = 0
    for word, weight in zip(keywords, weights):
        if isinstance(word, str) and isinstance(weight, int):
            score += text.count(word) * weight
        else:
            print("Error: Invalid data type detected in keywords or weights.")
    return score

app = Flask(__name__)

@app.route('/ranker', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resumes_file = request.files['resumes']
        keywords = request.form['keywords'].split(',')
        weights_str = request.form['weights'].split(',')
        # Convert weights to integers
        weights = [int(weight.strip()) for weight in weights_str]
        resumes_file.save("resume.csv")
        
        df = pd.read_csv('resume.csv')
        df.drop_duplicates(inplace=True)
        df['id'] = range(1, len(df) + 1)
        df.set_index('id', inplace=True)
        
        df['cleaned_text'] = df['Resume'].apply(clean_text)
        df['experience_score'] = df['cleaned_text'].apply(extract_experience)
        df['education_score'] = df['cleaned_text'].apply(extract_education)
        df['keyword_score'] = df['cleaned_text'].apply(lambda x: calculate_keyword_score(x, keywords, weights))
        
        weight_experience = 0.05
        weight_education = 0.10
        weight_keywords = 0.85
        
        df['rank_score'] = (weight_experience * df['experience_score']) + (weight_education * df['education_score']) + (weight_keywords * df['keyword_score'])
        
        df_sorted = df.sort_values(by='rank_score', ascending=False)
        resumes_data = df_sorted[['rank_score']].reset_index().to_dict('records')
        #resumes_data = df_sorted[['cleaned_text', 'rank_score']].head().to_dict('records')
        return render_template('index.html', resumes=resumes_data)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')