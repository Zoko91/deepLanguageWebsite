const recordButton = document.getElementById('record-button');
let stream;

recordButton.addEventListener('click', async () => {
  recordButton.style.display = 'none';


  // DEFINE A LOADER HERE


  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mediaRecorder = new MediaRecorder(stream);
  const chunks = [];

  mediaRecorder.start();

  setTimeout(() => {
    mediaRecorder.stop();
    stream.getTracks().forEach(track => track.stop());
  }, 5000);

  mediaRecorder.addEventListener('dataavailable', event => {
    chunks.push(event.data);
  });

  mediaRecorder.addEventListener('stop', () => {
    const blob = new Blob(chunks, { type: 'audio/mpeg' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'recording.mp3';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }, 100);
  });
});
