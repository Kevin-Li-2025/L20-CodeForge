# LCB Target Priority Analysis

This analysis compares public-signal-only targeting policies for the next candidate-aware behavior-test generation batch. Hidden-test outcomes are used only after the fact to measure target density; they are not inputs to the prompt builder or selector.

## Budget Comparison

| priority | budget | selected | hidden fail count | hidden fail rate | public pass count distribution |
| --- | ---: | ---: | ---: | ---: | --- |
| input-order | 32 | 32 | 2 | 0.0625 | `{"2": 6, "3": 9, "4": 17}` |
| input-order | 54 | 54 | 3 | 0.0556 | `{"2": 10, "3": 12, "4": 32}` |
| input-order | 64 | 64 | 3 | 0.0469 | `{"2": 11, "3": 13, "4": 40}` |
| input-order | 96 | 96 | 8 | 0.0833 | `{"2": 15, "3": 18, "4": 63}` |
| input-order | 128 | 128 | 15 | 0.1172 | `{"2": 23, "3": 22, "4": 83}` |
| input-order | 192 | 192 | 36 | 0.1875 | `{"2": 46, "3": 33, "4": 113}` |
| input-order | 256 | 256 | 55 | 0.2148 | `{"2": 63, "3": 46, "4": 147}` |
| public-fragility | 32 | 32 | 8 | 0.2500 | `{"2": 32}` |
| public-fragility | 54 | 54 | 16 | 0.2963 | `{"2": 54}` |
| public-fragility | 64 | 64 | 20 | 0.3125 | `{"2": 64}` |
| public-fragility | 96 | 96 | 34 | 0.3542 | `{"2": 96}` |
| public-fragility | 128 | 128 | 48 | 0.3750 | `{"2": 112, "3": 16}` |
| public-fragility | 192 | 192 | 70 | 0.3646 | `{"2": 112, "3": 80}` |
| public-fragility | 256 | 256 | 80 | 0.3125 | `{"2": 112, "3": 101, "4": 43}` |
| public-ambiguity | 32 | 32 | 1 | 0.0312 | `{"4": 32}` |
| public-ambiguity | 54 | 54 | 2 | 0.0370 | `{"4": 54}` |
| public-ambiguity | 64 | 64 | 2 | 0.0312 | `{"4": 64}` |
| public-ambiguity | 96 | 96 | 5 | 0.0521 | `{"4": 96}` |
| public-ambiguity | 128 | 128 | 12 | 0.0938 | `{"4": 128}` |
| public-ambiguity | 192 | 192 | 20 | 0.1042 | `{"4": 192}` |
| public-ambiguity | 256 | 256 | 38 | 0.1484 | `{"3": 53, "4": 203}` |

## Recommendation

`public-fragility` is the default next-batch targeter: it still uses only public scores, but prioritizes public-passing ties with fewer public-passing candidates and more partial public failures. That moves the behavior-test budget away from easy all-candidates-pass tasks and toward cases where candidate-aware verification can plausibly change the selected solution.
