# mfn

![example workflow](https://github.com/goromal/mfn/actions/workflows/test.yml/badge.svg)

Simple CLI tool meant to analyze an image of a single person and print whether the person appears to be a male (m), female (f), or neither (n). Uses vanilla OpenCV tools. Depending on the model, it can be pretty trigger-happy classifying genders even on inanimate objects, so for best results only use images of one person. Neural network model description and weights not included.

```bash
usage: mfn [Options] imgfile

Options:
  --model-proto arg     gender model description file
  --model-weights arg   gender model weights file
  --imgfile arg         image file to process
```
