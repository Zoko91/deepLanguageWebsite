const input = document.getElementById('formClick');
const loader = document.getElementById('loader');
const containerDiv = document.getElementById('cont');


input.addEventListener('click',  () => {
    input.style.display = "none";
    containerDiv.style.display = "flex";
});
