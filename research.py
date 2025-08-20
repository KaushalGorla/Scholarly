from flask import Flask, render_template, request

import openai
import random
import re

app = Flask(__name__)
openai.api_key = "example"

@app.route('/', methods=['GET', 'POST'])
def generate_article():
    if request.method == 'POST':
        topic = request.form['topic']
        intro_prompt = f"I want you to give me a brief background about {topic}. Write at least 400 words."
        intro_completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=intro_prompt,
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.5,
        )
        introduction = intro_completions.choices[0].text

        body_prompt = f"Talk about the future state of {topic}.Write at least 400 words."
        body_completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=body_prompt,
            max_tokens=3008,
            n=1,
            stop=None,
            temperature=0.5,
        )
        body_paragraphs = body_completions.choices[0].text

        conclusion_prompt = f"Talk about the current state of {topic}.Write at least 400 words."
        conclusion_completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=conclusion_prompt,
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.5,
        )
        conclusion = conclusion_completions.choices[0].text

        # Combine introduction, body paragraphs, and conclusion
        article = introduction + body_paragraphs + conclusion

        # Extract sources from the article
        sources = re.findall(r'(https?://[^\s]+)', article)
        if len(sources) > 5:
            sources = random.sample(sources, 5)

        # Render the form and the article
        return render_template('autoResearch.html', topic=topic, article=article, sources=sources)

    # Render the form
    return render_template('autoResearch.html')
    
if __name__ == '__main__':
    app.run()
