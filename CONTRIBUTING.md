# Contribution Guidelines

CLEASE team strongly welcomes all kinds of feedback, questions and contributions. Do not hesitate to open an issue on the GitLab page if you have questions or ideas for improvement.
When you open a merge request, we try to keep the history clean and concise with meaningful commit messages. We use the following prefixes for commits and merge request titles

* BUG: Fixed a runtime error, crash or wrong result
* DOC: Documentation changes
* ENH: Enhancement
* PERF: Performance improvement
* STYLE: Style improvements without any impact on logic (e.g., comments, indentations, PEP8, etc.)
* WIP: Work-in-progress not ready for merge

All commits will be squashed into one commit before it is merged. Thus, we encourage you
to split large contributions into smaller topic-based merge requests. This makes the review
process faster, and your code will be available as a part of the master branch sooner.

Further, the history will be cleaner if you rebase your changes to the tip of the master branch
before merging, although this is not a strict requirement. Here is a recipe of how you can do that on your command line.

1. Update the master branch to be [in sync](https://help.github.com/en/articles/syncing-a-fork) with the master branch in the main repo

```
git checkout master
git fetch upstream
git merge upstream/master
```

2. Checkout your feature branch
```
git checkout <branch-name>
```

3. Squash and rebase. A list with all the commits should appear.
```
git rebase -i master
```

4. Make sure the first commit says "pick", and replace the rest with "squash" or simply "s"

5. Save and close the text editor

6. A new editor is opened, where you have an opportunity to change the commit message. Craft a good commit message. For example ENH: Implements ...

7. Force push the final squashed commit
```
git push --force-with-lease origin
```

This recipe is based on the [wprig-project](https://github.com/wprig/wprig/wiki/How-to-squash-commits).

We encourage you to take part in the development, so don't hessitate to open MRs. CLEASE team will support you in the process of getting your ideas included in CLEASE!


## Code Style

Note, all of the packages discussed in the following section are installed automcatically if you included the `dev` dependencies when installing clease, e.g. with `pip install 'clease[dev]'`. As a developper, it is highly recommended that you have the `dev` tools installed in your environment.
### Pre commit
In the CLEASE code, we have a few code styles we adhere to. It is therefore recommended to use `pre-commit` hooks (for more information, please see the [pre-commit](https://pre-commit.com/) documentation) to check your commits before you push them to GitLab. If you have installed the `dev` CLEASE dependencies, you will have `pre-commit` installed, so simply run
```
pre-commit install
```
to set up the pre-commit hooks. These hooks will check any changed files for things like compliance with `flake8`, `black`, trailing white spaces and `pylint`.
When you make a new commit, the pre-commit hooks may apply changes to your file. If this is the case, simply add the changes to your staging area, and re-run the commit (or manually run the `pre-commit` command, prior to comitting to ensure everything passes).

### Flake8

First and foremost, we adhere to the [PEP8](https://pep8.org/) standards by requiring the code must comply with [flake8](https://flake8.pycqa.org/en/latest/) (barring a few exceptions, which are defined in the `.flake8` file, but these will be accounted for automatically by the `flake8` tool.) `flake8` compliance is checked by the GitLab CI, but can manually be checked on your local machine with (from the git source directory)
```
flake8 .
```

### Black
Next, we also require that the code adheres to the [black](https://black.readthedocs.io/en/stable/) code style. Black is a very strict auto-formatter. This is also checked in the pre-commit hooks, but can be manually applied with (from the git source directory)
```
black .
```
It might also be a good idea to set up `black` autoformatting in your IDE, e.g. VSCode, which will automatically apply these changes for you, while you develop. An example on how to set up VScode with `black` can be found [here](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0).

### Pylint

[pylint](https://pylint.org/) searches for possible mistakes in the code, and may start complaining about a lot of things. We have a rather extensive set of exceptions set up in the `.pylintrc` file, but it is still nevertheless quite restrictive on how to write good code. We require that pylint with our `.pylintrc` must pass in the GitLab CI, so therefore do your best to adhere to the pylint complaints. Only files in `clease` python source directory must comply with pylint, i.e. we do not require the `tests/` or `docs/` folder to comply.

 Pylint is also checked in the pre-commit hooks, but can be manually checked with (from the git root directory)
```
pylint clease/
```

### In case of issues
Do not be discouraged if it is difficult for you to comply with all of these code checks. If you cannot seem to manage, you can always do a commit with
```
git commit --no-verify
```
to circumvent the `pre-commit` checks. You can then push your code to GitLab, and requirest assistence from one of the other CLEASE developpers, and hopefully we can figure out how to make the code comply with our standards.
