'use strict';
const form = document.querySelector('#form');
const status = document.querySelector('#status');
const button = document.querySelector('#submit');
const seconds = document.querySelector('#seconds');
const output = document.querySelector('#results');
const pollMilliseconds = 1000;

async function json(response) {
  const value = await response.json();
  if (!response.ok) throw new Error(value.detail || 'Request failed.');
  return value;
}

fetch('/config').then(json).then(({limits}) => {
  seconds.max = limits.max_clip_seconds;
  seconds.value = Math.min(2, limits.max_clip_seconds);
  document.querySelector('#limits').textContent = `Up to ${limits.max_upload_mb} MiB per upload · ${limits.max_clip_seconds} seconds per job · one job at a time`;
}).catch(error => { status.textContent = error.message; });

form.addEventListener('submit', async event => {
  event.preventDefault();
  button.disabled = true;
  status.dataset.busy = 'true';
  status.dataset.error = 'false';
  output.replaceChildren();
  try {
    const data = new FormData();
    data.append('video', document.querySelector('#video').files[0]);
    status.textContent = 'Uploading video…';
    let job = await json(await fetch(`/jobs?max_seconds=${encodeURIComponent(seconds.value)}`, {method: 'POST', body: data}));
    status.textContent = 'Processing video…';
    while (job.status === 'running') {
      await new Promise(resolve => setTimeout(resolve, pollMilliseconds));
      job = await json(await fetch(`/jobs/${job.id}`));
    }
    if (job.status !== 'completed') throw new Error(job.error || 'Processing failed.');
    for (const [label, artifact] of [['Annotated video', 'video'], ['Summary JSON', 'result'], ['Observations JSONL', 'observations']]) {
      const link = document.createElement('a');
      link.textContent = label;
      link.href = `/jobs/${job.id}/${artifact}`;
      output.append(link);
    }
    const preview = document.createElement('video');
    preview.id = 'preview';
    preview.controls = true;
    preview.playsInline = true;
    preview.src = `/jobs/${job.id}/video`;
    output.append(preview);
    status.textContent = 'completed';
  } catch (error) {
    status.textContent = error.message;
    status.dataset.error = 'true';
  } finally {
    button.disabled = false;
    status.dataset.busy = 'false';
  }
});
