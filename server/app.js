const express = require('express');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const PORT = 5000;

// Serve static files from the "public" folder
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Endpoint to predict spam or ham
app.post('/predict', (req, res) => {
    const message = req.body.message || "";
    const pythonPath = `"C:\\Users\\thani\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"`;
    const command = `${pythonPath} model.py "${message}"`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            return res.status(500).json({ error: error.message });
        }
        if (stderr) {
            console.error(`stderr: ${stderr}`);
            return res.status(500).json({ error: stderr });
        }

        // Send the prediction result back to the client
        res.json({ prediction: stdout.trim() });
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
