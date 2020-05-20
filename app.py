import os
import pickle


import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.sparse
from dash.dependencies import Input, Output, State

from src.dash_helper import generate_table, fix_url
from src.model import cos_similarity, find_query_weights, most_similar

df = pd.read_csv("data/student-repos.csv")
tfid_vectorizer = pickle.load(open("data/model.pkl", "rb"))
X_train_weights = scipy.sparse.load_npz('data/model_sparse_matrix.npz')
date_file = open('data/last_refresh_date.txt',mode='r')
last_refresh_date = date_file.read()
date_file.close()

###########################################
# APP LAYOUT
###########################################

# COLOUR AND STYLE
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "UBC MDS GitHub Search"
port = int(os.environ.get("PORT", 5000))
server = app.server

colors = {
    "white": "#ffffff",
    "light_grey": "#d2d7df",
    "ubc_blue": "#082145",
    "secondary": "#95a5a6"
}

# DROP DOWN ITEMS
def create_dropdowns():
    file_types = df["file_extension"].unique().tolist()
    for i in range(0, len(file_types)-1):
        if type(file_types[i]) is float:
            file_types.pop(i)
    file_types = sorted(file_types, key=str.casefold)
    file_types = ["All"] + file_types
    repo_list = ["All"] + df["repo_name"].unique().tolist()
    out = {
        "file_types": file_types,
        "repo_list": repo_list
    }
    return out

# APP LAYOUT
app_navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("UBC MDS GitHub Search", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
    ],
    color="primary",
    dark=True,
)

app_search_filters = dbc.Row([
    dbc.Col([
        dcc.Slider(id="max_hits", min=1, max=20, step=1, value=5),
        dbc.Label("Max number of hits")
    ]),
    dbc.Col([
        dbc.Select(
            id="select_file_type", 
            options=[{"label": file_type , "value": file_type} for file_type in create_dropdowns()["file_types"]],
            value="All" 
        ),
        dbc.Label("Narrow search to specific file_type")
    ]),
    dbc.Col([
        dbc.Select(
            id="select_repo", 
            options=[{"label": repo , "value": repo} for repo in create_dropdowns()["repo_list"]],
            value="All"
        ),
        dbc.Label("Narrow search to specific repo")
    ])], align="end"
)

app_collapse_query_filters = html.Div(
    [
        dbc.Button(
            "Query Filters",
            id="collapse-button",
            className="mb-3",
            color="primary",
        ),
        dbc.Collapse(
            app_search_filters,
            id="collapse",
        ),
    ]
)

app_footer = dbc.Row([
    dbc.Col([
        html.P("Search database last updated: " + last_refresh_date),
        html.Br(),
        html.A("GitHub repo", href="https://github.com/SamEdwardes/ubc-mds-github-search"),
        html.P("Created by Sam Edwardes"),
        html.Br(),
        html.A("Icon my by Freepik from www.flaticon.com", href="https://www.flaticon.com/free-icon/seo_1055645?term=search&page=1&position=53")
    ])], align="start", justify="start"
)

app_query_plots = dbc.Row([
    dbc.Col([
        html.H5('Query Results'),
        html.Div(id='query_plot_subject')
    ])
])

app_main_body = dbc.Container(dbc.Row([
    dbc.Col(html.Div([
        html.Br(),
        html.P("Search the UBC MDS GitHub repository for specific items. You can access the repo here: "),
        html.A("https://github.ubc.ca/", href="https://github.ubc.ca/"),
        html.Br(),
        html.Br(),
        html.Label("Enter search term:"),
        html.Br(),
        dcc.Input(id="search_query", 
                  placeholder="e.g. maximum likelihood estimation",
                  type="text", size="75", value=""),
        html.Br(),
        html.Br(),
        app_collapse_query_filters,
        html.Br(),
        html.H5("Top hits"),
        html.Div(id="top_hits"),
        html.Br(),
        app_query_plots,
        html.Br(),
        html.Hr(),
        app_footer
    ]), width=10)
]))

app.layout = html.Div([
    app_navbar,
    app_main_body,
])

#/////////////////////////////////////////////////////////////////////////////
# APP CALL BACKS
#/////////////////////////////////////////////////////////////////////////////

@app.callback(
    Output(component_id='top_hits', component_property='children'),[
        Input(component_id='search_query', component_property='value'),
        Input(component_id='max_hits', component_property='value'),
        Input(component_id='select_file_type', component_property='value'),
        Input(component_id='select_repo', component_property='value')
    ]
)
def update_top_hits(search_query, max_hits, selected_file_type, selected_repo):
    # find most relevant panges
    X_query_weights = find_query_weights(search_query, tfid_vectorizer)
    sim_list = cos_similarity(X_query_weights, X_train_weights)
    df_out = df.copy()
    df_out["score"] = sim_list
    # filtering
    if selected_file_type == "All":
        df_out = df_out
    else:
        df_out = df_out[df_out["file_extension"] == selected_file_type]
    if selected_repo == "All":
        df_out = df_out
    else:
        df_out = df_out[df_out["repo_name"] == selected_repo]
    # cleaning df for viewing
    out = df_out.sort_values(by="score", ascending=False).head(max_hits)
    out["score"] = df_out["score"].round(2)
    out["size"] = (df_out["size"] * 1e-6).round(3) # convert bytes to megabytes
    out["url"] = out["url"].apply(fix_url)
    out["url"] = out["url"].apply(lambda x: html.A("link", href=x))
    out = out.drop(columns=["content", "content_clean", "encoding", "repo_full_name"])
    out = out.rename(columns={
        "repo_name": "Repo",
        "file_name": "File Name",
        "file_extension": "File Extension",
        "size": "Size (MB)",
        "path": "Path",
        "url": "URL",
        "score": "Score"
    })
    return dbc.Table.from_dataframe(out, bordered=True, hover=True, striped=True)


@app.callback(
    Output(component_id='query_plot_subject', component_property='children'),[
        Input(component_id='search_query', component_property='value')
    ]
)
def update_plot_subject(search_query):
    # find most relevant panges
    X_query_weights = find_query_weights(search_query, tfid_vectorizer)
    sim_list = cos_similarity(X_query_weights, X_train_weights)
    df_out = df.copy()
    df_out["score"] = sim_list
    # cleaning df for viewing
    df_out = df_out.query('score > 0')
    df_out = df_out[['repo_name', 'score']].groupby('repo_name').agg('sum')
    df_out = df_out.sort_values(by='score')
    df_out = df_out.reset_index().tail(10)
    
    fig = px.bar(
        df_out,
        x='score',
        y='repo_name',
        orientation='h',
        title='Most Relevant Repos'
    )

    fig.update_xaxes(title_text='Total Score')
    fig.update_yaxes(title_text='')

    return dcc.Graph(id='plot', figure=fig)


@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port=port
    )
