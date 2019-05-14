# NRP sequence classifier
NRP sequence-based predictor

A binary classifier based only on non-ribosomal peptide sequences to predict NRPs with anti-bacterial activity.

## Installation
This script uses Python 3.7.x. If you don't have Python, I would recommend downloading it from [Anaconda](https://www.continuum.io/downloads).

Copy or clone this package from Github.

Open the Terminal/Command Line and navigate to where you copied the package:

    $ cd path/to/copied/directory

### Linux and MacOS

Install the dependencies by entering:

    $ pip install -r requirements.txt

## Usage

To run conventional machine learning models from the command-line, just do:

    $ python conventional_models.py

Example: Example: Running tool with model_type=0

    $ python deep_learning_models.py -in path_to/trainingset_sequences.csv -o path_to_output/outputFolderName/ -mt 0

To run deep learning models from the command-line, just do:

    $ python deep_learning_models.py

Example: Example: Running tool with max_length=50 and embedding_length=32

    $ python deep_learning_models.py -in path_to/trainingset_sequences.csv -o path_to_output/outputFolderName/ -ml 50 -el 32

To list all the parameters needed from the command-line, just do:

    $ python conventional_models.py --help
    $ python deep_learning_models.py --help

## Questions and Comments

Feel free to direct any questions or comments to the Issues page of the repository.

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
