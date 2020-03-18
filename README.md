# UBC MDS GitHub Search

A tool for searching the UBC Masters of Data Science GitHub repository.

![screenshot-of-app](https://imgur.com/VxdPOVl.png)

## How to use

#### Running everything

To refresh the data, create the model, and run the app enter into the command line:

```
bash RUNALL.sh
```

Then open [http://0.0.0.0:5000/](http://0.0.0.0:5000/).

Note that for the script to work you will need to create an environment variable named `GITUBCTOKEN_ubc_mds_search`. This is a GitHub API token so that python can access the correct GitHub repos. You can create one by visiting [https://github.ubc.ca/settings/tokens](https://github.ubc.ca/settings/tokens), and then clicking on **Generate new token**. You should grant the following permissions:

![screenshot-token-permissions](https://imgur.com/k8vPMK0.png)

#### Running the app only

To just run the app enter the following into the command line:

```
python app.py
```

Then open [http://0.0.0.0:5000/](http://0.0.0.0:5000/).
