document.addEventListener('DOMContentLoaded', function () {
    var els = document.querySelectorAll("audio");
    for (var i = 0; i < els.length; i++) {
        console.log(els[i].src);
        let i_ = i;
        let src_ = els[i].src;
        let newNode = document.createElement("div")
        newNode.innerHTML = `<table style="padding-top:16px;padding-bottom:16px;">
                                <tr>
                                    <td style="position: relative;top: 4px;width:40px;">
                                        <div style="display:flex;justify-content: space-around;align-items: center;" class="controls">
                                            <button class="audiobtn" data-action="play" style="background: #fff; border: none; padding-left:0;padding-right:0;">
                                                <div class="playicon" style="width: 30px;height: 20px;background-color:green" >
                                                </div>
                                                <div class="pauseicon" style="display:none;width: 30px;height: 20px;background-color:red">
                                                </div>
                                            </button>
                                        </div>
                                    </td>
                                    <td width="100%">
                                        <div class="wave"></div>
                                    </td>
                                </tr>
                                </table>`;
        els[i_].parentNode.insertBefore(newNode, els[i_])
        els[i].remove();
        let wavesurfer = WaveSurfer.create({
            container: document.getElementsByClassName("wave")[i_],
            waveColor: '#1a1a1a',
            progressColor: '#ffffff',
            cursorColor: '#000000',
            backend: 'MediaElement',
            mediaControls: false,
            hideScrollbar: true,
            minPxPerSec: 120,
            normalize: true,
            height: 128,
            width: window.visualViewport.width*0.7
        });
        wavesurfer.once('ready', function () {
            console.log('Using wavesurfer.js ' + WaveSurfer.VERSION);
        });
        wavesurfer.on('error', function (e) {
            console.warn(e);
        });
        wavesurfer.on('play', function (e) {
            document.getElementsByClassName("playicon")[i_].style.display = "none";
            document.getElementsByClassName("pauseicon")[i_].style.display = "block";
        });
        wavesurfer.on('pause', function (e) {
            document.getElementsByClassName("playicon")[i_].style.display = "block";
            document.getElementsByClassName("pauseicon")[i_].style.display = "none";
        });
        newNode.querySelector('[data-action="play"]')
            .addEventListener('click', wavesurfer.playPause.bind(wavesurfer));
        wavesurfer.load(src_);
    }
});
