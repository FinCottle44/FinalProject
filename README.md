# CM3202 - Final Project

Evaluating and Extending a Repurposed Natural Language Machine Learning Model for Symbolic Music Completion

Created by Finlay Cottle of Cardiff University School of Computer Science and Informatics using [MusicBERT](https://pip.pypa.io/en/stable/) and the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).

## Requirements
- Linux Operating System (Project tested on Ubuntu 22.04.1 LTS)
- Python 3.7 (Virtual Environment creation recommended)
- CUDA 10 or later (Project tested on 12)

## Installation

Once requirements satisfied, use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Usage
For Experiment 1, using OctupleMIDI, we need to unzip the `lmd_full_data_bin.zip` file:
```bash
unzip lmd_full_data_bin.zip
```
Now, launch the project in Jupyter Lab and open Experiment `.ipynb` files:
```bash
jupyter lab
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
