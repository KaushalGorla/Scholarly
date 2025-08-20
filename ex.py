from flask import Flask, render_template, request
import openai
import re

app = Flask(__name__)
openai.api_key = "example"

app.jinja_env.globals.update(zip=zip)

def extract_terms_and_definitions(text):
    terms = []
    definitions = []
    chunk_size = 1000
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        prompt = f"Extract every single nouns, people, laws, and ideas and then add defenitions. Extract from the following text: '{chunk}'\n\nTerm:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=3024,
            n=1,
            stop=None,
            temperature=0.7,
        )
        result = response.choices[0].text.strip()
        term_definition_pairs = re.findall(r"Term:(.*?)\nDefinition:(.*?)\n", result, flags=re.DOTALL)
        for term_definition_pair in term_definition_pairs:
            term = term_definition_pair[0].strip()
            definition = term_definition_pair[1].strip()
            terms.append(term)
            definitions.append(definition)
        start += chunk_size
    return terms, definitions


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        terms, definitions = extract_terms_and_definitions(text)
        return render_template('Aiflash.html', terms=terms, definitions=definitions)
    else:
        return render_template('Aiflash.html')


if __name__ == '__main__':
    app.run(debug=True, port=6060)