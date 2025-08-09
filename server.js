const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const DATA_PATH = path.join(__dirname, 'public', 'data', 'combined_matches.json');

app.get('/api/games', (req, res) => {
  fs.readFile(DATA_PATH, 'utf8', (err, data) => {
    if (err) {
      res.status(500).json({ error: 'Data not found' });
    } else {
      res.setHeader('Content-Type', 'application/json');
      res.send(data);
    }
  });
});

app.use(express.static(path.join(__dirname, 'dist')));

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
