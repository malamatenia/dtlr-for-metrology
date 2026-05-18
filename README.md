# <p align="center"> Leveraging Morphology for Historical Script Metrological Analysis (ICDAR 2026) [![DOI](https://zenodo.org/badge/[add_badge].svg)](https://doi.org/10.5281/zenodo.18745702) <p/>

# <p align="center"> 🔗 [Project Webpage]([https://malamatenia.github.io/dtlr-for-metrology/]) </p> <p align="center"> <sub> [Malamatenia Vlachou Efstathiou](https://malamatenia.github.io/), [Raphael Baena](https://raphael-baena.github.io/), [Dominique Stutzmann](https://www.irht.cnrs.fr/fr/annuaire/stutzmann-dominique), [Mathieu Aubry](https://mathieuaubry.github.io/)</sub> </p>

![measures_definition](./media/measures_definition.jpeg)

## Purpose
 
This repository contains the scripts, notebook, and data needed to reproduce the metrological analysis of the paper. It also documents how to adapt the analysis to other datasets.

## Reproduce the paper
 
Open `metrological_analysis.ipynb` and run all cells. The notebook:
 
1. Builds the working corpus from the DTLR output in `input/`, applying the filters described in §3.2 of the paper (line-type, zone, error-neighbour, 4 σ outlier).
2. Computes every metric used in Figures 6–7: letter aspect ratios, bigram edge-to-edge distances, word distances.
3. Renders the linear and crossed plots into `results/`

For running the analysis code online, we provide a standalone Colab[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qqrAT2PDNlz-DvnFuEyUeYbUZT2frTKZ?usp=sharing) notebook.

## Run it on your own data
 
The analysis pipeline is intentionally modular: anything DTLR can produce, this code can analyse. Two steps:
 
**1. Generate DTLR outputs for your manuscript.** See the DTLR-for-paleography repository: [add url]. You will need:
 
- A folder of per-line prediction JSONs (one folder per document, one `.json` per line).
- A `transcribe.json` mapping character indices to characters.
- A folder of per-document character prototypes.

More details on how to apply the analysis in `metrological_analysis.ipynb`.


## Cite us

```bibtex
@misc{vlachou2026metrology,
    title = {Leveraging Morphology for Historical Script Metrological Analysis},
    author = {Vlachou-Efstathiou, Malamatenia and Baena, Raphael and Stutzmann, Dominique and Aubry, Mathieu},
    publisher = {Document Analysis and Recognition--ICDAR 2026 Vienna, Austria, August 30--September 4, 2026, Proceedings},
    year = {2026},
    organization={Springer}
```

Check out also: 
- [Baena, R., Kalleli, S., & Aubry, M. (2024). General Detection-based Text Line Recognition.](https://detection-based-text-line-recognition.github.io/)
- [Vlachou Efstathiou, M, Siglidis, I., Stutzmann, D., & Aubry, M. (2024). An Interpretable Deep Learning Approach for Morphological Script Type Analysis.](https://learnable-handwriter.github.io/)
- [Siglidis, I., Gonthier, N., Gaubil, J., Monnier, T., & Aubry, M. (2023). The Learnable Typewriter: A Generative Approach to Text Analysis.](https://imagine.enpc.fr/~siglidii/learnable-typewriter/)


## Acknowledgements
This study was supported by the CNRS through MITI and the 80|Prime program (CrEMe Caractérisation des écritures médiévales), and by the European Research Council (ERC project DISCOVER, number 101076028). We thank Sonat Baltacı, Syrine Kalleli, Marta Lopez-Rahut for valuable feedback on the paper.
