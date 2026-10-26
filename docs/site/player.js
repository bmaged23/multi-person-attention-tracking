'use strict';
const video = document.querySelector('#demo');
const play = document.querySelector('#play');
play.addEventListener('click', async () => {
  try {
    await video.play();
  } catch {
    play.hidden = false;
    video.controls = true;
  }
});
video.addEventListener('play', () => { play.hidden = true; });
video.addEventListener('ended', () => { play.hidden = false; });
