Contribution Guidelines
========================

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
git push -f origin
```

This recipe is based on the (wprig-project)[https://github.com/wprig/wprig/wiki/How-to-squash-commits]

We encourage you to take part in the development, so don't hessitate to open MRs. CLEASE team will support you in the process of getting your ideas included in CLEASE!