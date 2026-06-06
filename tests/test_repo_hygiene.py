from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_exploratory_notebooks_in_main_repo_surface():
    notebook_dir = ROOT / "notebooks"

    if not notebook_dir.exists():
        return

    assert not list(notebook_dir.glob("*.ipynb"))


def test_readme_exposes_open_source_project_signals():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required = [
        "EggHatch-AI is an open-source AI shopping agent prototype",
        "Quick Start",
        "Architecture",
        "Star History",
        "CONTRIBUTING.md",
        "LICENSE",
    ]

    for text in required:
        assert text in readme


def test_github_pages_entry_exists():
    page = ROOT / "docs" / "index.html"

    assert page.exists()
    content = page.read_text(encoding="utf-8")
    assert "<title>EggHatch-AI | AI Shopping Agent Prototype</title>" in content
