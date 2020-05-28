import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px

from src.dash_helper import fix_url
from src.model import cos_similarity, find_query_weights

colors = {
    "white": "#ffffff",
    "light_grey": "#d2d7df",
    "ubc_blue": "#082145",
    "secondary": "#95a5a6"
}


def perform_query(search_query, df, tfid_vectorizer, X_train_weights):
    """[summary]

    Parameters
    ----------
    search_query : [type]
        [description]
    df : [type]
        [description]
    tfid_vectorizer : [type]
        [description]
    X_train_weights : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    X_query_weights = find_query_weights(search_query, tfid_vectorizer)
    sim_list = cos_similarity(X_query_weights, X_train_weights)
    df_out = df.copy()
    df_out["score"] = sim_list
    df_out = df_out.query('score > 0')
    return df_out


def update_top_hits(query_results, max_hits, selected_file_type, selected_repo, df):
    """[summary]

    Parameters
    ----------
    query_results : [type]
        [description]
    max_hits : [type]
        [description]
    selected_file_type : [type]
        [description]
    selected_repo : [type]
        [description]
    df : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
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
    out["size"] = (df_out["size"] * 1e-6).round(3)  # convert bytes to megabytes
    out["url"] = out["url"].apply(fix_url)
    out['file_name'] = out['file_name'].apply(lambda x: x[0:25] + '...' + x[-25:] if len(x) > 50 else x)
    out["file_name"] = out.apply(lambda x: html.A(x['file_name'], href=str(x['url']), target='_blank'), axis=1)
    out = out.drop(columns=["encoding", "repo_full_name", 'path', 'url'])
    out = out.rename(columns={
        "repo_name": "Repo",
        "file_name": "File Name",
        "file_extension": "File Extension",
        "size": "Size (MB)",
        "url": "URL",
        "score": "Score"
    })
    return out


def update_plot_subject(query_results, df):
    """[summary]

    Parameters
    ----------
    query_results : [type]
        [description]
    df : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
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

    fig.update_traces(hovertemplate='<b>%{y}</b><br>Total Score: %{x}')
    fig.update_traces(marker_color=colors['ubc_blue'])
    fig.update_xaxes(title_text=None, showticklabels=False)
    fig.update_yaxes(title_text=None)
    fig['layout'].update(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig_out = dcc.Graph(id='plot', figure=fig, config={
        'displayModeBar': False
    })

    return fig_out


def update_number_hits(query_results, df):
    """[summary]

    Parameters
    ----------
    query_results : [type]
        [description]
    df : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
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


def update_number_subjects(query_results, df):
    """[summary]

    Parameters
    ----------
    query_results : [type]
        [description]
    df : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
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
