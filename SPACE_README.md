---
title: LungCanver Demo
emoji: 🫁
colorFrom: green
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# LungCanver — Demo Space

This Space runs a Gradio demo that downloads the fine-tuned ResNet50 model from the model repo
`Sarth001/LungCanver` and serves a simple image classification interface.

## Notes
- The app downloads the model at runtime using `hf_hub_download` (cached).
- If the build fails, open the Space page and check **Build logs**.

## Disclaimer
Research/educational use only — not for clinical diagnosis.