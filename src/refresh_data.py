import base64
import os
import re
from collections import defaultdict

import pandas as pd
from github import Github, Organization

from helpers import print_break
from model import text_preprocess


def get_repos():
    """Get all student repos from github and return as a list."""
    selected_repos = [
        "MDS-2019-20/DSCI_511_prog-dsci_students", 
        "MDS-2019-20/DSCI_521_platforms-dsci_students", 
        "MDS-2019-20/DSCI_542_comm-arg_students", 
        "MDS-2019-20/DSCI_551_stat-prob-dsci_students", 
        "MDS-2019-20/DSCI_512_alg-data-struct_students", 
        "MDS-2019-20/DSCI_523_data-wrangling_students", 
        "MDS-2019-20/DSCI_531_viz-1_students", 
        "MDS-2019-20/DSCI_552_stat-inf-1_students", 
        "MDS-2019-20/DSCI_571_sup-learn-1_students", 
        "MDS-2019-20/DSCI_561_regr-1_students", 
        "MDS-2019-20/DSCI_532_viz-2_students", 
        "MDS-2019-20/DSCI_513_database-data-retr_students", 
        "MDS-2019-20/DSCI_562_regr-2_students", 
        "MDS-2019-20/DSCI_573_feat-model-select_students", 
        "MDS-2019-20/DSCI_572_sup-learn-2_students", 
        "MDS-2019-20/DSCI_522_dsci-workflows_students", 
        "MDS-2019-20/DSCI_563_unsup-learn_students", 
        "MDS-2019-20/DSCI_524_collab-sw-dev_students", 
        "MDS-2019-20/DSCI_553_stat-inf-2_students", 
        "MDS-2019-20/DSCI_574_spat-temp-mod_students", 
        "MDS-2019-20/DSCI_575_adv-mach-learn_students", 
        "MDS-2019-20/DSCI_541_priv-eth-sec_students", 
        "MDS-2019-20/DSCI_525_web-cloud-comp_students", 
        "MDS-2019-20/DSCI_554_exper-causal-inf_students", 
        "MDS-2019-20/DSCI_591_capstone-proj_students"
    ]
    print_break("Getting repos from GitHub:")
    repo_list = []
    for i in selected_repos:
        try:
            repo_list.append(g.get_repo(i))
        except:
            print(f"{i} not found")
    return repo_list

def refresh_data(repo_list, max_repos=None):

    results = defaultdict(list)

    if max_repos is None:
        limit = len(repo_list)
    else:
        limit = max_repos

    # Iterate through each repo, and each file to extract content
    count = 1
    print_break("Getting content from each repo:")
    for repo in repo_list:
        print(repo)
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            # if a file is a directory, keeping digging deeper, otherwise start
            # to collect the metadata
            if file_content.type == "dir": 
                contents.extend(repo.get_contents(file_content.path))
            else:
                results["repo_name"].append(repo.name)
                results["repo_full_name"].append(repo.full_name)
                results["file_name"].append(file_content.name)
                results["file_extension"].append(os.path.splitext(file_content.name)[1])
                results["size"].append(file_content.size)
                results["path"].append(file_content.path)
                results["url"].append(file_content.url)
                # only get content from these file extensions
                file_extensions = r".*(\.md|\.py|\.Rmd|.ipynb)$"
                if file_content.size > 1000000 or not re.match(file_extensions, file_content.name):
                    results["encoding"].append("not read")
                    results["content"].append(file_content.name)
                else:                    
                    results["encoding"].append(file_content.encoding)
                    results["content"].append(base64.b64decode(file_content.content))        
        if count == limit:
            break
        else:
            count += 1

    return results

def decode_utf(x):
    """Helper function for decoding github content."""
    if type(x) == str:
        return x
    else:
        return x.decode("utf-8").lower()

def content_processing(content_dict):
    """Perform some basic preprocessing on conent from GitHub."""
    print_break("Performing preprocessing to content:")
    df = pd.DataFrame(data=content_dict)
    df["content"] = df.loc[:,"content"].apply(decode_utf)
    df["content_clean"] = df.loc[:,"content"].apply(text_preprocess)
    # df["file_extension"] = df["file_name"].apply(lambda x: os.path.splitext(x)[1])
    print("Preprocessing complete!")
    return df


if __name__ == "__main__":
    # Setup Github API
    token = os.environ.get("GITUBCTOKEN2")
    g = Github(base_url="https://github.ubc.ca/api/v3", login_or_token=token)
    # get content
    student_repos = get_repos()
    results = refresh_data(repo_list=student_repos, max_repos=2)
    df = content_processing(results)
    df.to_csv("data/2020-03-18_student-repos.csv", index=False)
