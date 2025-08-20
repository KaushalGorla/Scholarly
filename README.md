## Scholarly

An offline, API-free research and writing assistant powered by NLTK. Scholarly helps you generate structured background, current state, and future outlook sections for a topic using classical NLP techniques — no external API keys, network calls, or paid models.

### Why NLTK (and no APIs)?
- **Privacy-first**: All text processing runs locally; nothing leaves your machine.
- **Deterministic and explainable**: Classical NLP pipelines are transparent and tunable.
- **No keys, no quotas**: Works entirely without third-party services.

### What Scholarly does
- **Topic understanding**: Tokenization, POS tagging, stopword filtering, and WordNet-based synonym/definition lookups to expand a topic.
- **Outline building**: Simple heuristics over keywords, collocations, and noun-phrase chunks to propose section headings.
- **Draft generation (offline)**: Sentence templates stitched with corpus-derived n-grams and WordNet terms to create introductions, body paragraphs, and conclusions.
- **Enrichment**: Optional named-entity chunking, definition inserts, and quote extraction from local corpora.
- **Citations extraction**: Pulls URLs mentioned in text, if any, to present as sources.

Note: The UI currently renders a single form and a resulting article on the root route. The generation logic is designed to be implemented with NLTK, replacing any API-based calls.

### Project structure
- `research.py` — Flask app serving the main route and article generation.
- `templates/autoResearch.html` — UI for entering a topic and viewing the generated article.
- `templates/*.css/js` — Supporting assets.

### Requirements
- Python 3.9+
- Packages: `Flask`, `nltk`

### Installation
```bash
git clone <your-repo-url> scholarly
cd scholarly
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install Flask nltk

# Download required NLTK data (minimal set)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger')"

# For optional NER/chunking features
python -c "import nltk; nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

### Running the app
```bash
python research.py
# Visit http://127.0.0.1:5000/
```

### Using Scholarly
1. Open the app in your browser.
2. Enter a topic (e.g., "Quantum Computing").
3. Scholarly generates a structured article (intro, body, conclusion) and extracts any URLs present as sources.

### How the NLTK pipeline works (conceptual)
- **Preprocessing**: Sentence and word tokenization (`punkt`), lowercasing, stopword removal.
- **Keywording**: POS tagging to emphasize nouns/proper nouns; collocation detection for multi-word terms.
- **Expansion**: WordNet synonyms/definitions to vary language and add context.
- **Composition**: Template-based sentences blended with n-grams over a small local corpus you provide (e.g., notes or public-domain text).
- **Post-processing**: Simple frequency-based summarization and paragraph assembly.

You can customize these steps directly in the Flask route in `research.py`.

### Customization tips
- Add your own local corpus and build n-gram frequency tables with NLTK utilities to influence style and vocabulary.
- Tune stopword lists and POS filters to change the voice or density.
- Swap in different sentence templates per section for variety.

### Roadmap
- Pluggable strategies (e.g., switch between template-only vs. n-gram blend).
- Optional export to Markdown/PDF.
- In-app corpus management (upload local text, index, and reuse).

### Troubleshooting
- If tokenizers or taggers fail, ensure the NLTK data downloads completed successfully.
- For entirely offline use, make sure all required NLTK datasets are downloaded before disconnecting.

### License
Add a license of your choice (e.g., MIT) in a `LICENSE` file.
