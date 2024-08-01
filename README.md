# instance-based-loc

## Coding practices to follow

_Remove in the final code release_

1. Add an issue for every task that you'll be integrating
2. Create an appropriately named branch (or `dev-$USER` if no specific task) for all work
3. Create a pull request, link the issue that you're solving and then merge WHEN you're done adding the things mentioned in the issue.

Please do NOT push to main - we shouldn't have to rebase/other stuff for every few commits.

## Setup

### Setup conda environment

```bash
conda env create -f environment.yml
conda activate dator
```

_NOTE_: Update the environment YAML before merging any PR. Remove the `prefix` property from the YAML file as well. 

## Overall Documentation

Will be added
