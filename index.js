const express = require('express');
const app = express();
const path = require('path');
const multer = require('multer');

app.use(express.static(path.join(__dirname, 'public')));
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '/views'));

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.get('/', function (req, res) {
  res.render('index');
});

app.post('/predict', upload.single('image'), function (req, res) {
  // Process the uploaded image (req.file.buffer)
  // Replace the following with your actual image processing logic

  const resultData = {
    message: 'Image processed successfully!',
    // Include other result information if needed
  };

  res.json(resultData);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, function () {
  console.log(`Server is running on port ${PORT}`);
});
