# GitHub Pages demo

A static site with the existing **annotated classroom MP4**, a pipeline diagram,
measured-result summaries and links to setup/API/Docker/evaluation guides.
It does not run Python models or accept video uploads.

The Pages workflow stages this folder's HTML/CSS/JS, the 8MB annotated MP4,
its poster and the small pipeline SVG. It does not publish the 92MB GIF, private
uploads, weights, raw evaluation data or the complete documentation directory.
All video attribution is displayed directly below the player.

Local preview from the repository root:

```bash
python -m scripts.documentation.prepare_demo_site
python -m http.server 8780 --directory _site
```

Open http://localhost:8780. GitHub deployment URL:
https://bmaged23.github.io/multi-person-attention-tracking/
