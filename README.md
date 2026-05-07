# <p align="center"> Leveraging Morphology for Historical Script Metrological Analysis (ICDAR 2026) [![DOI](https://zenodo.org/badge/[add_badge].svg)](https://doi.org/10.5281/zenodo.18745702) <p/>

# <p align="center"> 🔗 [Project Webpage]([add_project_webpage]) </p> <p align="center"> <sub> [Malamatenia Vlachou Efstathiou](https://malamatenia.github.io/), [Raphael Baena](https://raphael-baena.github.io/), [Dominique Stutzmann](https://www.irht.cnrs.fr/fr/annuaire/stutzmann-dominique), [Mathieu Aubry](https://mathieuaubry.github.io/)</sub> </p>

![measures_definition.png](./.media/measures_definition.png)

### Purpose

This repository contains the scripts and data necessary for reproducing the results of the paper. We also detail how to use/adapt the script for your data. All data comes from the output of the DTLR-for-paleography method. On how to train the system, see : [add url of method] 


### Repository Structure

[add]

### Analysis scripts

For running the analysis code online, we provide a standalone Colab[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qqrAT2PDNlz-DvnFuEyUeYbUZT2frTKZ?usp=sharing) notebook.

## Run it on your data

What you'll need: 

From the DTLR model, adapted for paleographical analysis: [url to the method] output: 
- a folder with prototypes per letter
- a transcribe.json file with character indexing
- a prediction folder with .json metrics per line

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
