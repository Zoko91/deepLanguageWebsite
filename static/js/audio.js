const recordButton = document.getElementById('formClick');

let chunks = [];

recordButton.addEventListener('click', startRecording);

function startRecording() {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm',
        audioBitsPerSecond: 16000,
      });
      mediaRecorder.start();
      console.log('Recording started');

      setTimeout(() => {
        mediaRecorder.stop();
        console.log('Recording stopped');
      }, 5200);

      mediaRecorder.addEventListener('dataavailable', event => {
        chunks.push(event.data);
      });

      mediaRecorder.addEventListener('stop', () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');
        saveAudio(formData);
        chunks = [];
      });
    })
    .catch(error => {
      console.error(error);
    });
}

function saveAudio(formData) {
  fetch('/save_audio', {
    method: 'POST',
    body: formData,
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    console.log('Audio saved successfully');
    window.location.replace('/process');
  })
  .catch(error => {
    console.error('Error saving audio:', error);
  });
}
