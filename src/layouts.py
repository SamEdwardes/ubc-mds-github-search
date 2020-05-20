import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

date_file = open('data/last_refresh_date.txt',mode='r')
last_refresh_date = date_file.read()
date_file.close()

def app_navbar():
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


def create_dropdowns(df):
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


def app_collapse_query_filters(df):
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
                            options=[{"label": file_type , "value": file_type} for file_type in create_dropdowns(df)["file_types"]],
                            value="All" 
                        ),
                        dbc.Label("Narrow search to specific file_type")
                    ]),
                    dbc.Col([
                        dbc.Select(
                            id="select_repo", 
                            options=[{"label": repo , "value": repo} for repo in create_dropdowns(df)["repo_list"]],
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


def app_footer():
    layout = dbc.Row([
        dbc.Col([
            html.P("Search database last updated: " + last_refresh_date),
            html.Br(),
            html.A("GitHub repo", href="https://github.com/SamEdwardes/ubc-mds-github-search"),
            html.P("Created by Sam Edwardes"),
            html.Br(),
            html.A("Icon my by Freepik from www.flaticon.com", href="https://www.flaticon.com/free-icon/seo_1055645?term=search&page=1&position=53")
        ])], align="start", justify="start"
    )
    return layout