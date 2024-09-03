# CI/CD

## Actions

Create `~/.github/workflows/ci.yml`

```yml
name: ci 
on:
  push:
    branches:
      - main
jobs:
  deploy:
    runs-on: ubuntu-latest
    env:
      MKDOCS_GIT_COMMITTERS_APIKEY: ${{ secrets.GITHUB_TOKEN }}
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: '0'
      - uses: actions/setup-python@v3
        with:
          python-version: 3.x
          cache: pip
      - run: pip install -r requirements.txt
      - run: mkdocs gh-deploy --force --no-history
```

## DVC

Data Version Control

You can use an external storage for non-code files.

Especially useful for large files

