const recordButton = document.getElementById('record-button');
const containerDiv = document.getElementById('cont');
let stream;

recordButton.addEventListener('click', async () => {

    recordButton.style.display = 'none';
    containerDiv.style.display = 'flex';

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const mediaRecorder = new MediaRecorder(stream);
    const chunks = [];

    mediaRecorder.start();

    setTimeout(() => {
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
        containerDiv.style.display = 'none';

    }, 5000);

    mediaRecorder.addEventListener('dataavailable', event => {
        chunks.push(event.data);
    });

    mediaRecorder.addEventListener('stop', () => {
        const blob = new Blob(chunks, {type: 'audio/mpeg'});
        const formData = new FormData();
        formData.append('file', blob, 'recording.mp3');
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                window.location.replace('/process');
            })
            .catch(error => {
                console.error(error);
            });
    });
});
