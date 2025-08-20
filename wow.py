from flask import Flask, render_template, request
import openai
import re

# set up the OpenAI API
openai.api_key = "example"

# set up the Flask app
app = Flask(__name__)

# define a function to summarize the text and create bullet points
def summarize_text(text):
    # use the OpenAI API to summarize the text
    response = openai.Completion.create(
        engine="davinci",
        prompt=(f"Summarize the following text in bullet points:\n{text}\n\n-"),
        max_tokens=100
    )
    # extract the bullet points from the API response
    summary = response.choices[0].text
    bullet_points = re.findall(r'- (.+)', summary)
    return bullet_points

# define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    bullet_points = None
    if request.method == 'POST':
        # get the text from the form
        text = request.form['text']
        # summarize the text and create bullet points
        bullet_points = summarize_text(text)
    # render the template with the input form and bullet points (if available)
    return render_template('wow.html', bullet_points=bullet_points)

if __name__ == '__main__':
    app.run(debug=True, port=10000)
