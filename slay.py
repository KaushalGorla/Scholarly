from flask import Flask, render_template, request
import openai
import os

app = Flask(__name__)
openai.api_key = "your_api_key_here"  # Replace with your actual OpenAI API key

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['term']
        difficulty = request.form['difficulty']  # get the selected difficulty level
        prompt = ""

        # Set the prompt based on difficulty
        if difficulty == 'easy':
            prompt = f"Prepare a small test through the topic covered in this print. Remove some words. Students should be able to guess the answer unexpectedly by the subject matter presented in this print. Pose at least a few questions to test minimum knowledge about this subject. Start by understanding at least a few terms through this subject: {topic}. Add explanations to the end of each explanation, and each explanation should start from a new line."
        elif difficulty == 'moderate':
            prompt = f"Pose a question in Telugu related to the {topic} topic: "
        else:
            prompt = f"Add explanations to the end of each explanation, and each explanation should start from a new line. Create at least 10 questions. By the subject matter presented in this print, create many challenging questions through which students should be prepared to unexpectedly see answers. Students should pose questions to ascertain minimum knowledge about this subject. Start by understanding at least a few terms through this subject: {topic}. Add explanations to the end of each explanation, and each explanation should start from a new line."

        # Generate response using OpenAI
        essay = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000,
        )
        essay = essay.choices[0].text

        return render_template('actual.html', essay=essay)

    return render_template('actual.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3030))
    app.run(debug=True, port=port)
