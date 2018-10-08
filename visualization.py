#!/usr/bin/env python3
# coding: utf-8
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import networkx as nx
import igraph as ig
import os


def plot_3D(adj_list, df_node_label, title='Graph'):
    # Importing libs:

    # Copy to Newtwork x:
    graph1 = nx.Graph()
    graph1.add_edges_from(adj_list.values)
    N = graph1.number_of_nodes()
    L = graph1.number_of_edges()
    Edges = [tuple(i) for i in adj_list.values]
    G = ig.Graph(Edges, directed=True)
    # Node labels:
    group = df_node_label['label'].tolist()
    # Setting plotly
    layt = G.layout('kk', dim=3)

    Xn = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
    Yn = [layt[k][1] for k in range(N)]  # y-coordinates
    Zn = [layt[k][2] for k in range(N)]  # z-coordinates
    Xe = []
    Ye = []
    Ze = []
    for e in Edges:
        Xe += [layt[e[0]][0], layt[e[1]][0], None]  # x-coordinates of edge ends
        Ye += [layt[e[0]][1], layt[e[1]][1], None]
        Ze += [layt[e[0]][2], layt[e[1]][2], None]

        # PArameters:
    trace1 = go.Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          line=dict(color='rgb(125,125,125)', width=1),
                          hoverinfo='none'
                          )

    trace2 = go.Scatter3d(x=Xn,
                          y=Yn,
                          z=Zn,
                          mode='markers',
                          name='actors',
                          marker=dict(symbol='circle',
                                      size=6,
                                      color=group,
                                      colorscale='Viridis',
                                      line=dict(color='rgb(50,50,50)', width=0.5)
                                      ),
                          text=group,
                          hoverinfo='text'
                          )

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = go.Layout(
        title=title,
        width=1000,
        height=1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[
            dict(
                showarrow=False,
                text="Data source: {}".format(title),
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                    size=14
                )
            )
        ], )

    data = [trace1, trace2]
    #     plotly.offline.plot({'data': data, 'layout': layout},
    #              auto_open=True, image = 'png', image_filename='graph',
    #              output_type='file', image_width=800, image_height=600,
    #              filename='temp-plot.html', validate=False)

    dload = os.path.expanduser('~/Downloads')
    title_png = title + '.png'
    f_load = os.path.join(dload, title_png)
    f_save = os.path.join('/Users/marcelogutierrez/Projects/Gamma/capsuleSans/diagrams', title_png)
    html_file = '{}.html'.format(title)

    plotly.offline.plot(
        {"data": data,
         "layout": layout}, image='png', filename=html_file, image_filename=title, auto_open=True)

    sleep(3)

    shutil.move(f_load, f_save)