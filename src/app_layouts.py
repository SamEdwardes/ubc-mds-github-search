import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

date_file = open('data/last_refresh_date.txt', mode='r')
last_refresh_date = date_file.read()
date_file.close()

##############################################################################
# Helper functions
##############################################################################


def create_dropdowns(df):
    """[summary]

    Parameters
    ----------
    df : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
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


##############################################################################
# Layouts
##############################################################################


def app_navbar():
    """[summary]

    Returns
    -------
    [type]
        [description]
    """
    layout = dbc.Navbar(
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
    return layout


def app_collapse_query_filters(df):
    """[summary]

    Parameters
    ----------
    df : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    layout = html.Div(
        [
            dbc.Button(
                "Query Filters",
                id="collapse-button",
                className="mb-3",
                color="primary",
            ),
            dbc.Collapse(
                dbc.Row([
                    dbc.Col([
                        dcc.Slider(id="max_hits", min=1, max=20, step=1, value=5),
                        dbc.Label("Max number of hits")
                    ]),
                    dbc.Col([
                        dbc.Select(
                            id="select_file_type",
                            options=[{"label": file_type, "value": file_type} for file_type in create_dropdowns(df)["file_types"]],
                            value="All"
                        ),
                        dbc.Label("Narrow search to specific file_type")
                    ]),
                    dbc.Col([
                        dbc.Select(
                            id="select_repo",
                            options=[{"label": repo, "value": repo} for repo in create_dropdowns(df)["repo_list"]],
                            value="All"
                        ),
                        dbc.Label("Narrow search to specific repo")
                    ])], align="end"
                ),
                id="collapse",
            ),
        ]
    )
    return layout


def app_query_plots():
    """[summary]

    Returns
    -------
    [type]
        [description]
    """
    layout = dbc.Row([
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
    return layout


def app_footer():
    """Create app footer

    Returns
    -------
    [type]
        [description]
    """
    layout = dbc.Row([
        dbc.Col([
            html.B('Credits'),
            html.Br(),
            html.Br(),
            html.Em(['Created by ', html.A('Sam Edwardes', href="https://github.com/SamEdwardes/ubc-mds-github-search", target='_blank')]),
            html.Br(),
            html.Em(['Data from ', html.A('https://github.ubc.ca/', href="https://github.ubc.ca/", target='_blank')]),
            html.Br(),
            html.Em(['Icon by Freepik from ', html.A("www.flaticon.com", href="https://www.flaticon.com/free-icon/seo_1055645?term=search&page=1&position=53", target='_blank')]),
            html.Br(),
            html.Br()
        ]),
        dbc.Col([
            html.B('About'),
            html.Br(),
            html.Br(),
            html.Em("Last update: " + last_refresh_date),
            html.Br(),
            html.Em(['Created using ', html.A('Plotly Dash', href="https://dash.plotly.com/", target='_blank')]),
            html.Br(),
            html.Em(['Report a bug on ', html.A('GitHub Issues', href="https://github.com/SamEdwardes/ubc-mds-github-search/issues", target='_blank')]),
            html.Br(),
            html.Br()
        ])], align="start", justify="start"
    )
    return layout
