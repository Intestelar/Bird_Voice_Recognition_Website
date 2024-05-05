document.getElementById("startListeningBtn").addEventListener("click", function() {
    var listeningFeedback = document.getElementById("listeningFeedback");

    // Show listening feedback
    listeningFeedback.style.display = "block";

    // Check if the browser supports getUserMedia
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Use getUserMedia to capture audio from the user's microphone
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function(stream) {
                // Create a new MediaRecorder instance to record the audio stream
                var mediaRecorder = new MediaRecorder(stream);
                var chunks = [];

                // Start recording
                mediaRecorder.start();

                // Event handler for when data is available
                mediaRecorder.ondataavailable = function(event) {
                    chunks.push(event.data);
                };

                // Event handler for when recording is stopped
                mediaRecorder.onstop = function() {
                    // Convert the recorded audio data to a Blob
                    var audioBlob = new Blob(chunks, { type: 'audio/wav' });

                    // Create a FormData object to send the audio data to the backend
                    var formData = new FormData();
                    formData.append('audio_file', audioBlob);

                    // Send the audio data to the Flask backend using fetch API
                    fetch('/submit', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Handle the prediction result
                        console.log(data);
                        // Redirect or display prediction result as needed
                    })
                    .catch(error => {
                        // Handle errors
                        console.error('Error:', error);
                    });
                };

                // Stop recording after 5 seconds
                setTimeout(function() {
                    // Stop recording
                    mediaRecorder.stop();
                    // Hide listening feedback
                    listeningFeedback.style.display = "none";
                }, 5000); // Change the time as needed (in milliseconds)
            })
            .catch(function(err) {
                // Handle errors
                console.error('Error capturing audio: ' + err);
            });
    } else {
        // Browser does not support getUserMedia
        console.error('getUserMedia not supported on your browser');
    }
});
