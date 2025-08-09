# Table Hockey H2H

A modern React dashboard for exploring head‑to‑head results in table hockey tournaments. Data is loaded from static JSON files generated from parquet sources.

## Setup

```bash
npm install
python scripts/convert_parquet_to_json.py  # generates JSON in public/data/
npm run dev
```

## Build & Serve

```bash
npm run build            # build static assets to dist/
npm start                # serve dist/ with /api/games endpoint
```

## Testing

```bash
npm test
```

## Data preprocessing

Parquet files live in the `data/` directory. Convert them to JSON before running the app using the provided script. The JSON files are ignored by git.

## Deployment

The project is built with [Vite](https://vitejs.dev/) and can be deployed as static files to Netlify, Vercel or GitHub Pages. When hosting statically, ensure the generated JSON files in `public/data/` are uploaded alongside the build output.
