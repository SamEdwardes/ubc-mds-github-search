import os
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import scipy.sparse
from dash.dependencies import Input, Output

from src.dash_helper import generate_table, fix_url
from src.model import cos_similarity, find_query_weights, most_similar

df = pd.read_csv("data/2020-03-18_student-repos.csv")
tfid_vectorizer = pickle.load(open("data/model.pkl", "rb"))
X_train_weights = scipy.sparse.load_npz('data/model_sparse_matrix.npz')

###########################################
# APP LAYOUT
###########################################

# COLOUR AND STYLE
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "UBC MDS GitHub Search"
port = int(os.environ.get("PORT", 5000))
server = app.server

colors = {"white": "#ffffff",
          "light_grey": "#d2d7df",
          "ubc_blue": "#082145"
          }

# APP LAYOUT
app.layout = html.Div(style={'backgroundColor': colors['light_grey']}, children=[
    # HEADER
    html.Div(className="row", style={'backgroundColor': colors['ubc_blue'], "padding": 10}, children=[
        html.H2('UBC MDS GitHub Search',
                style={'color': colors['white']})
    ]),
    # MAIN BODY
    html.Div(className="row", children=[
        # SIDEBAR
        html.Div(className="two columns", style={'padding': 20}, children=[
            html.A("GitHub Repo", href="https://github.com/SamEdwardes/github-search")
        ]),
        # SEARCH
        html.Div(className="ten columns", style={"backgroundColor": colors['white'], "padding": 20}, children=[
            html.H4("Search GitHub Repos"),
            html.Label("Enter search term"),
            dcc.Input(id="search_query", placeholder="e.g. maximum likelihood estimation",
                      type="text", size="75", value=""),
            html.Br(),
            html.Br(),
            html.Label("Max number of hits:"),
            dcc.Slider(id="max_hits", min=1, max=20, step=1, value=5),
            html.Br(),
            html.Hr(),
            html.H5("Top hits"),
            html.Div(id="top_hits")
        ])
    ])
])

###########################################
# APP CALL BACKS
###########################################

@app.callback(
    Output(component_id='top_hits', component_property='children'),
    [Input(component_id='search_query', component_property='value'),
     Input(component_id='max_hits', component_property='value')]
)
def update_top_hits(search_query, max_hits):
    X_query_weights = find_query_weights(search_query, tfid_vectorizer)
    sim_list = cos_similarity(X_query_weights, X_train_weights)
    df["score"] = sim_list
    out = df.sort_values(by="score", ascending=False).head(max_hits)
    out["url"] = out["url"].apply(fix_url)
    out["url"] = out["url"].apply(lambda x: html.A("link", href=x))
    out = out.drop(columns=["content", "content_clean"])
    return generate_table(out)


if __name__ == '__main__':
    app.run_server(debug=True,
                   host="0.0.0.0",
                   port=port)
