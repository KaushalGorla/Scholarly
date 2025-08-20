from flask import Flask, render_template, request
import openai
import re

app = Flask(__name__)
openai.api_key = "example"

@app.route('/', methods=['GET', 'POST'])
def generate_article():
    if request.method == 'POST':
        topic = request.form['topic']
        intro_prompt = f"Write a detailed summary of the following information that makes sense: {topic}"
        intro_completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=intro_prompt,
            max_tokens=2500,
            n=1,
            stop=None,
            temperature=0.5,
        )
        introduction = intro_completions.choices[0].text
        
        
        bullet_prompt = f"Create multiple bullet points based on the following summary and the star bullet point for example, use this * not numbers or dashes:<br>{introduction}"
        bullet_completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=bullet_prompt,
            max_tokens=2500,
            n=1,
            stop=None,
            temperature=0.5,
        )
        bullet_points = bullet_completions.choices[0].text
        bullet_points = re.sub(r'\*(?!\d)', '<br> • ', bullet_points)

        # Combine introduction, bullet points, and conclusion
        article = f"{introduction}<br><br>{bullet_points}"
        return render_template('burp.html', topic=topic, article=article)

    
    return render_template('burp.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
