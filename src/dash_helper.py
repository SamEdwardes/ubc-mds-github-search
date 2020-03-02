import pandas as pd
import dash_html_components as html
import re

def generate_table(df, max_rows=10):
    """
    Renders a table in dash app
    Arguments:
        df {pd.DataFrame} -- Data frame to render
    Keyword Arguments:
        max_rows {int} -- number of rows to render (default: {10})
    Returns:
        html.Table -- table ready to be rendered by
    """
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +
        # Body
        [html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), max_rows))]
    )

def fix_url(x):

    bad = "https://github.ubc.ca/api/v3/repos/MDS-2019-20/DSCI_552_stat-inf-1_students/contents/lectures/08_lecture-maximum-likelihood-estimation.ipynb?ref=master"
    good = "https://github.ubc.ca/MDS-2019-20/DSCI_552_stat-inf-1_students/blob/master/lectures/08_lecture-maximum-likelihood-estimation.ipynb"
    
    # regex = re.compile('[^a-zA-Z ]')
    # x = regex.sub('', x)
    x = x.replace("api/v3/repos/", "")
    x = x.replace("/contents/", "/blob/master/")
    x = x.replace("?ref=master", "")
    return x

def test_fix_url():
    bad = "https://github.ubc.ca/api/v3/repos/MDS-2019-20/DSCI_552_stat-inf-1_students/contents/lectures/08_lecture-maximum-likelihood-estimation.ipynb?ref=master"
    good = "https://github.ubc.ca/MDS-2019-20/DSCI_552_stat-inf-1_students/blob/master/lectures/08_lecture-maximum-likelihood-estimation.ipynb"
    test = fix_url(bad)
    print(good)
    print(test)
    assert test == good