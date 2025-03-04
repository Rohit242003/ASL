// script.js

const video = document.getElementById('video-feed');
const textField = document.getElementById('text-field');
const suggestionsDiv = document.getElementById('suggestions');

// Access the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing webcam:', err);
    });

// Function to send video frames to the backend for prediction
function predictASL() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg');

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData, sentence: textField.value }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction) {
            textField.value += data.prediction;
            speakText(data.prediction);
        }
        if (data.suggestions && data.suggestions.length > 0) {
            // Clear previous suggestions
            suggestionsDiv.innerHTML = "";
            // Add buttons for each suggestion
            data.suggestions.forEach(word => {
                const button = document.createElement('button');
                button.textContent = word;
                button.classList.add('btn', 'btn-outline-secondary', 'suggestion-button');
                button.onclick = () => {
                    textField.value += ' ' + word; // Insert the word into the textbox
                };
                suggestionsDiv.appendChild(button);
            });
            suggestionsDiv.classList.add('fade-in'); // Add fade-in animation
        } else {
            suggestionsDiv.innerHTML = "<p class='text-muted'>No suggestions available</p>";
        }
    })
    .catch(error => {
        console.error('Error predicting ASL:', error);
    });
}

// Predict ASL every 1 second
setInterval(predictASL, 1000);

// Function to clear the text field
function clearText() {
    textField.value = '';
    suggestionsDiv.innerHTML = "<p class='text-muted'>Suggestions will appear here</p>";
}

// Function to change voice to female
function changeVoice() {
    fetch('/speak', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: 'Changing voice to female' }),
    });
}

// Function to speak the text
let isSpeaking = false;

function speakText(text = null) {
    if (isSpeaking) {
        console.log("Already speaking. Please wait.");
        return;
    }

    const textToSpeak = text || textField.value;
    if (!textToSpeak) {
        console.error("No text to speak.");
        return;
    }

    isSpeaking = true;
    console.log("Sending text to speak:", textToSpeak);
    fetch('/speak', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: textToSpeak }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log("Text spoken successfully.");
        } else {
            console.error("Error speaking text:", data.message);
        }
        isSpeaking = false;  // Reset the flag
    })
    .catch(error => {
        console.error('Error speaking text:', error);
        isSpeaking = false;  // Reset the flag
    });
}
