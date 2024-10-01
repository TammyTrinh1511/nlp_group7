# Pre-commit Hook Setup

This repository uses [pre-commit](https://pre-commit.com/) to ensure code quality and consistency before every commit. Pre-commit hooks automatically run checks like linting, formatting, and testing on the code before it's committed.

Extension Trailing Spaces -> detect trailing spaces code-server
## Installation

### 1. Install Pre-commit

To install `pre-commit`, first ensure you have Python and `pip` installed, then run the following command:

```bash
pip install pre-commit
```
### 2. Install the Git Hooks

Once `pre-commit` is installed, run the following command to install the pre-commit hooks into your local Git repository:

```bash
pre-commit install
```
### 3. Running Pre-commit Hooks
Once installed, the hooks will run automatically every time you make a commit. If any issues are found, the commit will be rejected until they are fixed.

After finish coding. To manually run the pre-commit hooks on all files, use:
```bash
pre-commit run --all-files
```
