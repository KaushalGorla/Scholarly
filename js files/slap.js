// Initialize media recorder
let mediaRecorder;

// Initialize audio element
let audio = document.querySelector('#recording');

// Initialize record button
let recordButton = document.querySelector('#record');

// Initialize stop button
let stopButton = document.querySelector('#stop');

// Initialize recordings list
let recordingsList = document.querySelector('#recordings');

// Start recording
function startRecording() {
  // Request permission to use microphone
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    // Initialize media recorder
    mediaRecorder = new MediaRecorder(stream);
    
    // Start recording
    mediaRecorder.start();
    
    // Enable stop button
    stopButton.disabled = false;
    
    // Change record button text
    recordButton.innerHTML = 'Recording...';
  });
}

// Stop recording
function stopRecording() {
  // Stop media recorder
  mediaRecorder.stop();
  
  // Disable stop button
  stopButton.disabled = true;
  
  // Change record button text
  recordButton.innerHTML = 'Record';
}

// Handle data available event
mediaRecorder.ondataavailable = e => {
  // Create audio element
  let audioElement = document.createElement('audio');
  
  // Set audio source
  audioElement.src = URL.createObjectURL(e.data);
  
  // Set audio controls
  audioElement.controls = true;
  
  // Create list item
  let listItem = document.createElement('li');
  
  // Append audio element to list item
  listItem.appendChild(audioElement);
  
  // Append list item to recordings list
  recordingsList.appendChild(listItem);
};
