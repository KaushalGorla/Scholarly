import openai
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

openai.api_key = "example"


text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A quick brown dog jumps over the lazy fox."


model = "text-davinci-002"
prompt_settings = {
    'model': model,
    'prompt': '',
    'max_tokens': 0,
    'temperature': 0,
}
encoder = openai.Model.list(prompt_settings=prompt_settings)['data'][0]['id']
text1_encoding = openai.Completion.create(model=encoder, prompt=text1, max_tokens=1024, temperature=0)
text2_encoding = openai.Completion.create(model=encoder, prompt=text2, max_tokens=1024, temperature=0)


text1_vector = json.loads(text1_encoding.choices[0].text)
text2_vector = json.loads(text2_encoding.choices[0].text)


similarity = cosine_similarity([text1_vector], [text2_vector])[0][0]


print(f"The similarity between the two texts is: {similarity}")
