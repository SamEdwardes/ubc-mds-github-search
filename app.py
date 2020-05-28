import os
import pickle

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import scipy.sparse
from dash.dependencies import Input, Output, State

from src.app_callbacks import (perform_query, update_number_hits,
                               update_number_subjects, update_plot_subject,
                               update_top_hits)
from src.app_layouts import (app_collapse_query_filters, app_footer,
                             app_navbar, app_query_plots)

cols_to_read = ['repo_name', 'repo_full_name', 'file_name', 'file_extension', 'size', 'path', 'url', 'encoding']
df = pd.read_csv("data/student-repos.csv", usecols=cols_to_read)
tfid_vectorizer = pickle.load(open("data/model.pkl", "rb"))
X_train_weights = scipy.sparse.load_npz('data/model_sparse_matrix.npz')


##############################################################################
# APP STYLES
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


##############################################################################
# APP LAYOUT
##############################################################################

welcome_text = html.P([
    'This app was created to help students from the University of British Columbia ',
    html.A('(UBC)', href="https://www.ubc.ca/", target='_blank'),
    ' Master of Data Science ',
    html.A('(MDS)', href="https://masterdatascience.ubc.ca/", target='_blank'),
    ' program to help find course conent.',
    '  Anyone can use the app, but the links will only work for those who have access to the ',
    html.A('MDS GitHub repo', href="https://github.ubc.ca/", target='_blank'),
    '.'
])

app_main_body = dbc.Container(dbc.Row([
    dbc.Col(html.Div([
        html.Br(),
        welcome_text,
        html.Br(),
        html.Label("Enter search term:"),
        html.Br(),
        dcc.Input(id="search_query", placeholder="e.g. maximum likelihood estimation", type="text", size="75", value=""),
        html.Br(),
        html.Br(),
        app_collapse_query_filters(df),
        html.Br(),
        html.H5("Top hits"),
        html.Div(id="top_hits"),
        html.Br(),
        app_query_plots(),
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
    Output(component_id='query_results', component_property='children'),
    [Input(component_id='search_query', component_property='value')]
)
def callback_perform_query(search_query):
    df_out = perform_query(search_query, df, tfid_vectorizer, X_train_weights)
    return df_out.to_json(orient='split')


# Step 2a: Create a dataframe out of the n most likely results
@app.callback(
    Output(component_id='top_hits', component_property='children'), [
        Input(component_id='query_results', component_property='children'),
        Input(component_id='max_hits', component_property='value'),
        Input(component_id='select_file_type', component_property='value'),
        Input(component_id='select_repo', component_property='value')]
)
def callback_update_top_hits(query_results, max_hits, selected_file_type, selected_repo):
    df_out = update_top_hits(query_results, max_hits, selected_file_type, selected_repo, df)
    return dbc.Table.from_dataframe(df_out, bordered=True, hover=True, striped=True, size='md')


# Step 2b: Create a plot of the most relevant subjects
@app.callback(
    Output(component_id='query_plot_subject', component_property='children'),
    [Input(component_id='query_results', component_property='children')]
)
def callback_update_plot_subject(query_results):
    fig_out = update_plot_subject(query_results, df)
    return fig_out


# Step 2c: Count number of hits
@app.callback(
    Output(component_id='number_of_hits', component_property='children'),
    [Input(component_id='query_results', component_property='children')]
)
def callback_update_number_hits(query_results):
    num_hits_label = update_number_hits(query_results, df)    
    return num_hits_label


# Step 2d: Count number of subjects
@app.callback(
    Output(component_id='number_of_subjects', component_property='children'),
    [Input(component_id='query_results', component_property='children')]
)
def callback_update_number_subjects(query_results):
    num_subjects_label = update_number_subjects(query_results, df)
    return num_subjects_label


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
