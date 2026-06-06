# Contributing to EggHatch-AI

Thanks for taking a look at EggHatch-AI. This project is a compact AI agent prototype, so the best contributions are focused, easy to review, and honest about the proof-of-concept scope.

## Good First Contributions

- improve README clarity or setup instructions
- add small product/review fixtures
- add lightweight tests for existing modules
- improve recommendation explanations
- add examples of useful shopping queries
- implement one bounded adapter, such as benchmark lookup or price parsing

## Development Setup

```bash
git clone https://github.com/AustinZ21/EggHatch-AI.git
cd EggHatch-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

Run the lightweight checks:

```bash
pytest
```

Runtime dependencies for the full Streamlit demo are in `requirements.txt`.

## Pull Request Guidelines

- Keep PRs small and focused.
- Include tests for behavior changes when possible.
- Avoid committing generated model files, downloaded weights, notebook outputs, local `.env` files, or caches.
- Make limitations explicit when adding new agent behavior.
- Do not add live scraping or API calls without clear rate-limit and failure behavior.

## Project Scope

EggHatch-AI is a local open-source prototype. It is not a production ecommerce platform, does not provide financial advice, and is not affiliated with Newegg.
