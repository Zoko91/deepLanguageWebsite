const input = document.getElementById('formClick');
const loader = document.getElementById('loader');
const containerDiv = document.getElementById('cont');


input.addEventListener('click',  () => {
    input.style.display = "none";
    containerDiv.style.display = "flex";
});
//
// let typed = new Typed('#typewriter', {
//     strings: ['You have 5 seconds to record yourself once you press the magic button'],
//     typeSpeed: 35,
//     showCursor: false
// });
