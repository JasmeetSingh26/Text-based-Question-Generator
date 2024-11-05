import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sense2vec import Sense2Vec
import spacy
import pandas as pd

# Initialize T5 model and tokenizer for question generation
tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

# Load SpaCy and Sense2Vec
nlp = spacy.load("en_core_web_sm")
s2v = Sense2Vec().from_disk('s2v_reddit_2015_md/s2v_old')

# Function to get important keywords
def get_imp_keywords(content):
    return [token.text for token in nlp(content) if token.pos_ in [ "PROPN", "NUM"]]

# Function to generate a question
def get_question(context, answer):
    prompt = f"answer: {answer}  context: {context} </s>"
    encoding = tokenizer.encode_plus(prompt, max_length=256, truncation=True, return_tensors="pt")
    
    outs = model.generate(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"],
                          early_stopping=True, num_beams=5, num_return_sequences=1, max_length=300)
    return tokenizer.decode(outs[0], skip_special_tokens=True).replace("question:", "").strip()

# Function to get similar words using Sense2Vec
def sense2vec_get_words(word):
    word = word.lower().replace(" ", "_")
    sense = s2v.get_best_sense(word)
    if not sense:
        return []
    return [each_word[0].split("|")[0].replace("_", " ").title() for each_word in s2v.most_similar(sense, n=20) if each_word[0].split("|")[0] != word]

# Generate multiple questions and return as DataFrame
def generate_questions(context, num_questions=5):
    questions_data = []
    for idx, answer in enumerate(get_imp_keywords(context)):
        distractors = sense2vec_get_words(answer.capitalize())
        if distractors:
            question = get_question(context, answer)
            options = [answer.capitalize()] + distractors[:3]
            question_dict = {
                "Question": question,
                "Answer": answer.capitalize(),
                "Option 1": options[0],
                "Option 2": options[1] if len(options) > 1 else "",
                "Option 3": options[2] if len(options) > 2 else "",
                "Option 4": options[3] if len(options) > 3 else ""
            }
            questions_data.append(question_dict)
        if idx >= num_questions - 1:
            break
    return pd.DataFrame(questions_data)

# Streamlit Interface
st.title("Text-based Question Generator")
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Enter text content in the text area below.
2. Set the number of questions you want to generate.
3. Click "Generate Questions" to view the questions.
4. Use the download button to save questions as a CSV file.
""")

# Text input for user to enter content
text_input = st.text_area("Enter text content...", height=200)

# Adjustable number of questions
num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=5)

if st.button("Generate Questions") and text_input:
    with st.spinner("Generating questions..."):
        questions_df = generate_questions(text_input, num_questions)
        st.write("Generated Questions")
        
        # Display each question in an accordion-style format
        for i, row in questions_df.iterrows():
            with st.expander(f"Question {i + 1}"):
                st.write(f"**Question:** {row['Question']}")
                st.write(f"**Answer:** {row['Answer']}")
                st.write("**Options:**")
                for j in range(1, 5):
                    option = row[f"Option {j}"]
                    if option:
                        st.write(f"{j}. {option}")

        # Add a CSV download button
        csv = questions_df.to_csv(index=False)
        st.download_button(
            label="Download Questions as CSV",
            data=csv,
            file_name="generated_questions.csv",
            mime="text/csv"
        )
