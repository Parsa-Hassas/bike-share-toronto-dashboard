# Bike Share Toronto Management Dashboard

Interactive Dash application for historical Bike Share Toronto ridership, station segments, first-model forecasts, and operational pressure indicators.

## Validate the package

Open PowerShell in this folder and run:

```powershell
py -3 validate_deployment.py
```

Expected result:

```text
DEPLOYMENT PACKAGE STATUS: PASS
```

## Run locally

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
py -3 app.py
```

Open `http://127.0.0.1:8050` in a browser. Stop the server with `Ctrl+C`.

## Publish with GitHub and Render

1. Extract this folder.
2. In GitHub Desktop, choose **File > Add Local Repository** and select this folder.
3. If prompted, create a repository here.
4. Commit all files and publish the repository to GitHub.
5. In Render, choose **New > Web Service** and connect the GitHub repository.
6. Use these settings:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Branch | `main` |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app:server --workers 1 --timeout 120` |
| Instance | Free or your preferred paid plan |

7. Select **Create Web Service**. Render will provide the public URL after deployment succeeds.

The included `render.yaml` contains the same service configuration.

## Important notes

- The repository must retain the current folder structure.
- The cached dashboard datasets are included; the host does not need the original 25-million-row input files.
- Do not upload passwords, API keys, `.env` files, or private source datasets.
- On Render's free service, the first request after an idle period may load slowly.
