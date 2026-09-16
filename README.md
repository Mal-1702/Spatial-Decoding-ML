# Are spatial auditory-attention decoders reading the eyes?

A gaze-confound analysis of EEG-based spatial auditory attention decoding on the
KU Leuven **AV-GC-AAD** dataset (Rotaru et al., 2024, *J. Neural Eng.* 21:016017;
Zenodo 10.5281/zenodo.11058711).

## Question

EEG decoders that predict *which direction* a listener attends often report high
accuracy. People usually look where they listen, and eye movements create large
electrical artifacts on the scalp. So is the decoder reading attention, or eye
position?

AV-GC-AAD lets us test this, because its four conditions change the relation between
gaze and attention:

| Condition | What the listener looks at | Gaze vs attended side |
|---|---|---|
| FixedVideo | video of the attended speaker, on the attended side | **confounded** |
| NoVisuals | black screen, central fixation | minimised |
| MovingVideo | video of the attended speaker moving randomly | **decorrelated** |
| MovingTargetNoise | cross-hair moving randomly, with background noise | **decorrelated** |

Each trial is 10 min. At 5 min the two speakers **swap sides**. The listener keeps
attending the same speaker, so the attended direction flips halfway through every
trial.
