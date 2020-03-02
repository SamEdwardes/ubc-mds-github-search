import base64
import os
import pandas as pd
import re

from collections import defaultdict
from github import Github, Organization

from model import text_preprocess

# using username and password
token = os.environ.get("GITUBCTOKEN2")
# Github Enterprise with custom hostname
g = Github(base_url="https://github.ubc.ca/api/v3", login_or_token=token)
student_repos = g.search_repositories(query="students in:repo:owner/name+pushed>2020-02-09+user:MDS-2019-20")

def refresh_data(max_repos):

    results = defaultdict(list)
    file_extensions = r".*(\.md|\.py|\.Rmd|.ipynb)$"
    limit = max_repos

    # get all content
    count = 1
    for repo in student_repos:
        print(repo)
        if "MDS-2019-20" not in repo.full_name:
            print("Not MDS 2019-20")
            continue
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
    #             print(file_content)
    #             print(file_content.name)
                results["repo_name"].append(repo.name)
                results["repo_full_name"].append(repo.full_name)
                results["file_name"].append(file_content.name)
                results["size"].append(file_content.size)
                results["path"].append(file_content.path)
                results["url"].append(file_content.url)
                
                if file_content.size > 1000000 or not re.match(file_extensions, file_content.name):
                    results["encoding"].append("not read")
                    results["content"].append(file_content.name)
                    continue
                    
                results["encoding"].append(file_content.encoding)
                results["content"].append(base64.b64decode(file_content.content))
                
        if count > limit:
            break
        else:
            count += 1

    return results

def decode_utf(x):
    if type(x) == str:
        return x
    else:
        return x.decode("utf-8").lower()

results = refresh_data(max_repos=20)
df = pd.DataFrame(data=results)
df["content"] = df.loc[:,"content"].apply(decode_utf)
df["content_clean"] = df.loc[:,"content"].apply(text_preprocess)
# TODO: add function to get extension from file name
df.to_csv("data/2020-03-01_student-repos.csv", index=False)
