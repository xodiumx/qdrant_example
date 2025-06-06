# Example of work with vector db Qdrant

## Instalation

- Clone repository

- Run db in docker:

```sh
make qd
```

- Create venv and install dependencies

```sh
uv venv
```

```sh
uv sync --all-groups
```

## Notebooks:

In derictory notebooks example of work

- Fill in the db use `./notebooks/crud.ipynb`

- Try search with embeddings use `./notebooks/search_with_embeddings.ipynb`
