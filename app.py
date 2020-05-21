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
from src.layouts import app_navbar, app_collapse_query_filters, app_footer

cols_to_read = ['repo_name', 'repo_full_name', 'file_name', 'file_extension', 
                'size', 'path', 'url', 'encoding']
df = pd.read_csv("data/student-repos.csv", usecols=cols_to_read)
tfid_vectorizer = pickle.load(open("data/model.pkl", "rb"))
X_train_weights = scipy.sparse.load_npz('data/model_sparse_matrix.npz')


##############################################################################
# APP LAYOUT
##############################################################################

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

app_query_plots = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader('Most relevant repos'),
            dbc.CardBody(html.Div(id='query_plot_subject'))
        ])
    ]),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader('Number of hits'),
            dbc.CardBody(html.Div(id='number_of_hits'))
        ]),
        html.Br(),
        dbc.Card([
            dbc.CardHeader('Number of subjects'),
            dbc.CardBody(html.Div(id='number_of_subjects'))
        ])
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
        app_collapse_query_filters(df),
        html.Br(),
        html.H5("Top hits"),
        html.Div(id="top_hits"),
        html.Br(),
        app_query_plots,
        html.Br(),
        html.Hr(),
        app_footer(),
        # Hidden div inside the app that stores the intermediate value
        html.Div(id='query_results', style={'display': 'none'})
    ]), width=10)
]))

app.layout = html.Div([
    app_navbar(),
    app_main_body,
])

##############################################################################
# APP CALL BACKS
##############################################################################

# Step 1: Perform the query and find the most similar results
@app.callback(
    Output(component_id='query_results', component_property='children'),[
        Input(component_id='search_query', component_property='value')]
)
def perform_query(search_query):
    X_query_weights = find_query_weights(search_query, tfid_vectorizer)
    sim_list = cos_similarity(X_query_weights, X_train_weights)
    df_out = df.copy()
    df_out["score"] = sim_list
    df_out = df_out.query('score > 0')

    return df_out.to_json(orient='split')


# Step 2: Create a dataframe out of the n most likely results
@app.callback(
    Output(component_id='top_hits', component_property='children'),[
        Input(component_id='query_results', component_property='children'),
        Input(component_id='max_hits', component_property='value'),
        Input(component_id='select_file_type', component_property='value'),
        Input(component_id='select_repo', component_property='value')]
)
def update_top_hits(query_results, max_hits, selected_file_type, selected_repo):
    # get query results
    df_out = pd.read_json(query_results, orient='split')
    
    # return a placeholder if there is no search input
    if df_out.shape[0] == 0:
        df_out = df.copy()
        df_out['score'] = np.NaN

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
    # out["url"] = out["url"].apply(lambda x: html.A("link", href=x))
    out['file_name'] = out['file_name'].apply(lambda x: x[0:25] + '...' + x[-25:] if len(x) > 50 else x) 
    out["file_name"] = out.apply(lambda x: html.A(x['file_name'], href=str(x['url'])), axis=1)
    out = out.drop(columns=["encoding", "repo_full_name", 'path', 'url'])
    out = out.rename(columns={
        "repo_name": "Repo",
        "file_name": "File Name",
        "file_extension": "File Extension",
        "size": "Size (MB)",
        "url": "URL",
        "score": "Score"
    })
    return dbc.Table.from_dataframe(out, bordered=True, hover=True, 
                                    striped=True, size='md')


# Step 3: Create a plot of the most relevant subjects
@app.callback(
    Output(component_id='query_plot_subject', component_property='children'),[
        Input(component_id='query_results', component_property='children')]
)
def update_plot_subject(query_results):
    # get query results
    df_out = pd.read_json(query_results, orient='split')

    # return a placeholder if there is no search input
    if df_out.shape[0] == 0:
        df_out = pd.DataFrame({
            'repo_name': df['repo_name'].unique()
        })
        df_out['score'] = 1
        df_out.reset_index().tail(10)
    else:
        df_out = df_out.query('score > 0')
        df_out = df_out[['repo_name', 'score']].groupby('repo_name').agg('sum')
        df_out = df_out.sort_values(by='score')
        df_out = df_out.reset_index().tail(10)

    def clean_string(x):
        x = x.lower()
        x = x.replace('_students', '')
        x = x.replace('dsci_', '')
        x = x.replace('_', ' ')
        x = x.replace('-', ' ')
        return x

    df_out['repo_name'] = df_out['repo_name'].apply(clean_string)
    df_out['score'] = df_out['score'].round(2)
    
    fig = px.bar(
        df_out,
        x='score',
        y='repo_name',
        orientation='h'
    )

    fig.update_traces(hovertemplate='<b>%{y}</b><br>Total Score: %{x}') # 
    fig.update_traces(marker_color=colors['ubc_blue'])
    fig.update_xaxes(title_text=None, showticklabels=False)
    fig.update_yaxes(title_text=None)
    fig['layout'].update(margin=dict(l=0,r=0,b=0,t=0))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig_out = dcc.Graph(id='plot', figure=fig, config={
        'displayModeBar': False
    })

    return fig_out


# Step 4: Count number of hits
@app.callback(
    Output(component_id='number_of_hits', component_property='children'),[
        Input(component_id='query_results', component_property='children')]
)
def update_number_hits(query_results):
    df_out = pd.read_json(query_results, orient='split')
    total_documents = df.shape[0]
    # return a placeholder if there is no search input
    if df_out.shape[0] == 0:
        total_hits = total_documents
    else:
        df_out = df_out.query('score > 0')
        total_hits = df_out.shape[0]

    out = dbc.Label([f'{total_hits:,} / {total_documents:,}', 
                    html.Br(), 
                    f'{(total_hits/total_documents*100):.0f}%'])
    
    return out


# Step 5: Count number of subjects
@app.callback(
    Output(component_id='number_of_subjects', component_property='children'),[
        Input(component_id='query_results', component_property='children')]
)
def update_number_subjects(query_results):
    total_subjects = df['repo_name'].unique().shape[0]
    df_out = pd.read_json(query_results, orient='split')

    # return a placeholder if there is no search input
    if df_out.shape[0] == 0:
        num_subjects = total_subjects
    else:
        df_out = df_out.query('score > 0')
        num_subjects = df_out['repo_name'].unique().shape[0]

    out = dbc.Label([f'{num_subjects:,} / {total_subjects:,}', 
                    html.Br(), 
                    f'{(num_subjects/total_subjects*100):.0f}%'])
    
    return out

# Other callbacks
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
