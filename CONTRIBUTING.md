# Contribution Guidelines
Thank you for your interest in contributing to the Lance Deep Learning Recipes repository! Your efforts help make this resource more valuable for the entire deep learning community using Lance.

This document will outline some basic guidelines for the contribution process. We're excited to see you contribute!

## Getting Started
- Fork the repository on GitHub.
- Clone your fork locally: `git clone https://github.com/your-username/lance-deeplearning-recipes.git`
- Create a new branch for your contribution: `git checkout -b your-feature-branch`

## Types of Contributions
We encourage the following types of contributions:

- New deep learning examples using Lance file format (either for datasets or artefact management)
- Improvements to existing examples
- Documentation enhancements
- Bug fixes and performance optimizations

## Adding a New Example
To add a new deep learning example:
- Create a new directory in the appropriate section (Dataset Examples or Training Examples).
- Include all necessary scripts, notebooks, and a [README.md](https://github.com/lancedb/lance-deeplearning-recipes/blob/main/README.md) file explaining the example.
- Update the main [README.md](https://github.com/lancedb/lance-deeplearning-recipes/blob/main/README.md) to include your new example in the relevant table.

## Code Style and Guidelines
- Follow PEP 8 guidelines for Python code.
- Use clear and descriptive variable names.
- Include comments to explain complex operations.
- Ensure your code is compatible with the latest stable version of Lance.

## Submitting Your Contribution
- Please ensure your pull request adheres to the following guidelines:
    - Make an individual pull request for each example you wish to add.
    - Link to the corresponding issue in your PR message (if previously discussed over an issue).

- Commit your changes 
    - If you are committing any form of addition, the message should have `add:` prefix. For example: `git commit -am "add: a new cool use-case"`
    - If you are committing any form of change or fix, the message should have `fix:` prefix. For example: `git commit -am "fix: this really irritating bug"`

- Push to your fork: `git push origin your-feature-branch`
- Open a pull request from your fork to the main repository.
- In the pull request description, explain your changes and their purpose.

## Review Process
A maintainer will review your pull request. They may request changes or ask questions. Once approved, your contribution will be merged.

## Community Guidelines
Be respectful and inclusive in all interactions and provide constructive feedback on other contributions!